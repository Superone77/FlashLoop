#!/usr/bin/env python3
"""PyTorch reference semantics for Ouro prefill LoopFreeze."""

from __future__ import annotations

import math
import types
from typing import Any, Callable

import torch
import torch.nn.functional as F


def cumulative_exit_probability(hazards: torch.Tensor) -> torch.Tensor:
    """Convert sequential exit hazards into cumulative probability of exiting."""
    if hazards.ndim < 1 or hazards.shape[-1] <= 0:
        raise ValueError("hazards must have a non-empty loop dimension")
    if torch.any((hazards < 0) | (hazards > 1)):
        raise ValueError("exit hazards must be probabilities")
    survival = torch.cumprod(1.0 - hazards.float(), dim=-1)
    return 1.0 - survival


def select_active_mask(
    scores: torch.Tensor,
    *,
    fraction: float,
    eligible: torch.Tensor | None = None,
    force_last_token: bool = True,
) -> torch.Tensor:
    """Select high-score tokens, optionally nested inside an eligible mask."""
    if scores.ndim != 2 or scores.shape[-1] <= 0:
        raise ValueError("scores must have shape [batch, tokens]")
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must satisfy 0 < fraction <= 1")
    if eligible is None:
        eligible = torch.ones_like(scores, dtype=torch.bool)
    if eligible.shape != scores.shape or eligible.dtype != torch.bool:
        raise ValueError("eligible must be a boolean mask matching scores")
    batch, tokens = scores.shape
    requested = max(1, math.ceil(tokens * fraction))
    result = torch.zeros_like(eligible)
    for batch_index in range(batch):
        candidates = torch.nonzero(eligible[batch_index], as_tuple=False).flatten()
        if candidates.numel() == 0:
            raise ValueError("eligible mask contains an empty batch row")
        count = min(requested, int(candidates.numel()))
        forced: list[int] = []
        if force_last_token:
            last = tokens - 1
            if not bool(eligible[batch_index, last]):
                raise ValueError("last token must remain eligible when it is forced")
            forced = [last]
            candidates = candidates[candidates != last]
        remaining = max(0, count - len(forced))
        if remaining:
            local = torch.topk(scores[batch_index, candidates].float(), remaining).indices
            result[batch_index, candidates[local]] = True
        if forced:
            result[batch_index, forced] = True
    return result


def carry_frozen_hidden(
    previous: torch.Tensor, computed: torch.Tensor, active: torch.Tensor
) -> torch.Tensor:
    """Return the computed hidden only at active prompt positions."""
    if previous.shape != computed.shape or active.shape != previous.shape[:-1]:
        raise ValueError("hidden and active-mask shapes differ")
    return torch.where(active.unsqueeze(-1), computed, previous)


def replace_frozen_cache(
    previous: torch.Tensor, current: torch.Tensor, active: torch.Tensor
) -> torch.Tensor:
    """Mix post-RoPE K/V cache rows using the adjacent loop as the proxy."""
    if previous.shape != current.shape or current.ndim != 4:
        raise ValueError("cache tensors must have equal [batch, heads, tokens, dim] shapes")
    if active.shape != (current.shape[0], current.shape[2]):
        raise ValueError("active mask does not match cache token positions")
    return torch.where(active[:, None, :, None], current, previous)


def active_only_final_norm(
    hidden: torch.Tensor,
    norm: Callable[[torch.Tensor], torch.Tensor],
    active: torch.Tensor,
) -> torch.Tensor:
    """Apply the loop-final norm only to active positions."""
    normalized = norm(hidden)
    return carry_frozen_hidden(hidden, normalized, active)


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return query * cos + _rotate_half(query) * sin, key * cos + _rotate_half(key) * sin


def _repeat_kv(values: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return values
    batch, heads, tokens, dim = values.shape
    return values[:, :, None].expand(batch, heads, groups, tokens, dim).reshape(
        batch, heads * groups, tokens, dim
    )


class LoopFreezeIntervention:
    """Reference LoopFreeze closure for Ouro prefill Loops 3/4.

    The implementation still computes dense projections and layer bodies, then discards
    frozen rows. Counters and fractions therefore describe algorithmic work opportunity,
    not wall-clock acceleration.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        fraction_loop3: float,
        fraction_loop4: float,
        selector: str = "hidden_delta",
        explicit_masks: dict[int, torch.Tensor] | None = None,
        random_seed: int = 0,
    ) -> None:
        if not 0.0 < fraction_loop4 <= fraction_loop3 <= 1.0:
            raise ValueError("LoopFreeze requires 0 < loop4 <= loop3 <= 1")
        if selector not in {"hidden_delta", "random", "explicit"}:
            raise ValueError("unsupported LoopFreeze selector")
        if selector == "explicit" and explicit_masks is None:
            raise ValueError("explicit selector requires masks for loops 3 and 4")
        self.model = model
        self.backbone = model.model
        self.num_layers = int(model.config.num_hidden_layers)
        self.fractions = {2: float(fraction_loop3), 3: float(fraction_loop4)}
        self.selector = selector
        self.explicit_masks = explicit_masks or {}
        self.generator = torch.Generator(device="cpu").manual_seed(int(random_seed))
        self.loop_hidden: dict[int, torch.Tensor] = {}
        self.active_masks: dict[int, torch.Tensor] = {}
        self.current_loop = 0
        self.current_is_prefill = False
        self.layer_originals: list[Any] = []
        self.attention_originals: list[Any] = []
        self.norm_original: Any | None = None
        self.cache_replacements = 0
        self.frozen_layer_rows = 0
        self.dense_target_calls = 0
        self._install()

    @staticmethod
    def _is_prefill(hidden_states: torch.Tensor, cache_position: torch.Tensor | None) -> bool:
        if hidden_states.shape[-2] != 1:
            return True
        return bool(cache_position is not None and cache_position.numel() == 1 and int(cache_position.item()) == 0)

    def _install(self) -> None:
        for layer_index, layer in enumerate(self.backbone.layers):
            self._install_attention(layer_index, layer.self_attn)
            self._install_layer(layer_index, layer)
        self._install_norm(self.backbone.norm)

    def _install_attention(self, layer_index: int, module: torch.nn.Module) -> None:
        original = module.forward
        self.attention_originals.append(original)

        def wrapped(attention_module, hidden_states, position_embeddings, attention_mask,
                    past_key_value=None, cache_position=None, current_ut=0, **kwargs):
            current_loop = int(current_ut)
            self.current_loop = current_loop
            self.current_is_prefill = self._is_prefill(hidden_states, cache_position)
            if not self.current_is_prefill or current_loop < 2:
                return original(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    cache_position=cache_position,
                    current_ut=current_loop,
                    **kwargs,
                )
            self.dense_target_calls += 1
            return self._completed_attention(
                layer_index, attention_module, hidden_states, position_embeddings,
                attention_mask, past_key_value, cache_position, current_loop
            )

        module.forward = types.MethodType(wrapped, module)

    def _completed_attention(
        self,
        layer_index: int,
        module: torch.nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        cache: Any,
        cache_position: torch.Tensor | None,
        current_loop: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cache is None or current_loop not in self.active_masks:
            raise RuntimeError("LoopFreeze target attention is missing cache or active mask")
        batch, tokens, _ = hidden_states.shape
        head_dim = int(module.head_dim)
        query_heads = int(module.config.num_attention_heads)
        kv_heads = int(module.config.num_key_value_heads)
        query = module.q_proj(hidden_states).view(batch, tokens, query_heads, head_dim).transpose(1, 2)
        key = module.k_proj(hidden_states).view(batch, tokens, kv_heads, head_dim).transpose(1, 2)
        value = module.v_proj(hidden_states).view(batch, tokens, kv_heads, head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        query, key = _apply_rope(query, key, cos, sin)
        cache_index = current_loop * self.num_layers + layer_index
        key, value = cache.update(
            key, value, cache_index,
            {"sin": sin, "cos": cos, "cache_position": cache_position},
        )
        previous_index = (current_loop - 1) * self.num_layers + layer_index
        previous_key = cache.key_cache[previous_index]
        previous_value = cache.value_cache[previous_index]
        if previous_key is None or previous_value is None:
            raise RuntimeError("LoopFreeze adjacent cache slot is missing")
        active = self.active_masks[current_loop]
        key = replace_frozen_cache(previous_key, key, active)
        value = replace_frozen_cache(previous_value, value, active)
        cache.key_cache[cache_index] = key
        cache.value_cache[cache_index] = value
        self.cache_replacements += 1
        repeated_key = _repeat_kv(key, int(module.num_key_value_groups))
        repeated_value = _repeat_kv(value, int(module.num_key_value_groups))
        logits = torch.matmul(query, repeated_key.transpose(2, 3)) * float(module.scaling)
        if attention_mask is not None:
            logits = logits + attention_mask[:, :, :, : repeated_key.shape[-2]]
        weights = F.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
        output = torch.matmul(weights, repeated_value).transpose(1, 2).contiguous()
        output = module.o_proj(output.reshape(batch, tokens, -1))
        return output, weights

    def _install_layer(self, layer_index: int, module: torch.nn.Module) -> None:
        original = module.forward
        self.layer_originals.append(original)

        def wrapped(_layer, hidden_states, *args, **kwargs):
            current_loop = int(kwargs.get("current_ut", 0))
            is_prefill = self._is_prefill(hidden_states, kwargs.get("cache_position"))
            output = original(hidden_states, *args, **kwargs)
            if is_prefill and current_loop >= 2:
                active = self.active_masks[current_loop]
                self.frozen_layer_rows += int((~active).sum().item())
                output = carry_frozen_hidden(hidden_states, output, active)
            return output

        module.forward = types.MethodType(wrapped, module)

    def _install_norm(self, module: torch.nn.Module) -> None:
        original = module.forward
        self.norm_original = original

        def wrapped(_norm, hidden_states):
            current_loop = self.current_loop
            if not self.current_is_prefill:
                return original(hidden_states)
            output = original(hidden_states)
            if current_loop >= 2:
                output = carry_frozen_hidden(hidden_states, output, self.active_masks[current_loop])
            self.loop_hidden[current_loop] = output.detach()
            if current_loop == 1:
                self.active_masks[2] = self._choose_mask(2, output)
            elif current_loop == 2:
                self.active_masks[3] = self._choose_mask(3, output)
            return output

        module.forward = types.MethodType(wrapped, module)

    def _choose_mask(self, target_loop: int, current_hidden: torch.Tensor) -> torch.Tensor:
        if target_loop in self.explicit_masks:
            mask = self.explicit_masks[target_loop].to(current_hidden.device)
            if mask.shape != current_hidden.shape[:-1]:
                raise ValueError("explicit LoopFreeze mask shape mismatch")
            return mask.bool()
        previous_hidden = self.loop_hidden[target_loop - 2]
        eligible = self.active_masks.get(target_loop - 1)
        if self.selector == "hidden_delta":
            scores = torch.linalg.vector_norm(current_hidden.float() - previous_hidden.float(), dim=-1)
            scores = scores / torch.linalg.vector_norm(current_hidden.float(), dim=-1).clamp_min(1e-12)
        elif self.selector == "random":
            scores = torch.rand(current_hidden.shape[:-1], generator=self.generator).to(current_hidden.device)
        else:
            raise RuntimeError("explicit LoopFreeze mask was not supplied")
        return select_active_mask(
            scores,
            fraction=self.fractions[target_loop],
            eligible=eligible,
            force_last_token=True,
        )

    def audit(self) -> dict[str, Any]:
        return {
            "active_counts": {str(loop + 1): int(mask.sum().item()) for loop, mask in self.active_masks.items()},
            "cache_replacements": self.cache_replacements,
            "frozen_layer_rows": self.frozen_layer_rows,
            "dense_target_calls": self.dense_target_calls,
            "mask_nested": bool(
                2 in self.active_masks
                and 3 in self.active_masks
                and torch.all(~self.active_masks[3] | self.active_masks[2]).item()
            ),
        }

    def close(self) -> None:
        for layer, original_layer, original_attention in zip(
            self.backbone.layers, self.layer_originals, self.attention_originals, strict=True
        ):
            layer.forward = original_layer
            layer.self_attn.forward = original_attention
        if self.norm_original is not None:
            self.backbone.norm.forward = self.norm_original

