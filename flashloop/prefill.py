"""Real compact-row prefill execution for late Ouro loops.

Unlike the earlier LoopFreeze diagnostic, this module never sends inactive prompt
rows through Q/K/V projections, the output projection, or the MLP.  A single
transient full composite K/V state is updated in place semantically (returned as a
new owner) so active queries can still attend every causal prompt position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .weights import fused_mlp, project_qkv


@dataclass(frozen=True)
class PrefillCompositeCache:
    """One full transient K/V version per physical layer during prefill only."""

    keys: list[torch.Tensor]
    values: list[torch.Tensor]

    def __post_init__(self) -> None:
        if len(self.keys) != len(self.values):
            raise ValueError("composite K/V layer counts differ")
        for key, value in zip(self.keys, self.values, strict=True):
            if key.shape != value.shape or key.ndim != 4 or key.shape[0] != 1:
                raise ValueError("composite K/V entries must match [1, heads, tokens, dim]")


@dataclass(frozen=True)
class PrefillLayerUpdate:
    """Post-RoPE K/V rows produced at one physical layer."""

    layer_index: int
    positions: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class PrefillReceipt:
    """Auditable proof that late-loop work scales with active prompt rows."""

    loop: int
    total_rows: int
    active_rows: int
    qkv_rows: int
    output_projection_rows: int
    mlp_rows: int
    dense_target_calls: int
    sparse_target_calls: int
    original_positions: tuple[int, ...]


@dataclass(frozen=True)
class PrefillLoopOutput:
    hidden_states: torch.Tensor
    composite_cache: PrefillCompositeCache
    updates: tuple[PrefillLayerUpdate, ...]
    receipt: PrefillReceipt


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (
        query * cos + _rotate_half(query) * sin,
        key * cos + _rotate_half(key) * sin,
    )


def _repeat_kv(values: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return values
    batch, heads, tokens, head_dim = values.shape
    return values[:, :, None].expand(batch, heads, groups, tokens, head_dim).reshape(
        batch,
        heads * groups,
        tokens,
        head_dim,
    )


def _validate_common(
    layers: Sequence[nn.Module],
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
) -> tuple[int, int]:
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError("the first prefill engine requires hidden shape [1, tokens, hidden]")
    if not layers:
        raise ValueError("prefill requires at least one physical layer")
    tokens = int(hidden_states.shape[1])
    cos, sin = position_embeddings
    if cos.shape != sin.shape or cos.ndim != 3 or cos.shape[:2] != (1, tokens):
        raise ValueError("position embeddings must match [1, tokens, head_dim]")
    if attention_mask is not None:
        if attention_mask.ndim != 4 or attention_mask.shape[0] != 1:
            raise ValueError("attention_mask must be additive [1, heads|1, queries, keys]")
        if attention_mask.shape[-2:] != (tokens, tokens):
            raise ValueError("prefill attention mask must cover the full prompt")
    return tokens, int(hidden_states.shape[-1])


def _execute_prefill_loop(
    layers: Sequence[nn.Module],
    hidden_states: torch.Tensor,
    *,
    active_mask: torch.Tensor,
    composite_cache: PrefillCompositeCache | None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    final_norm: nn.Module,
    loop: int,
    sparse: bool,
    consume_inputs: bool,
    retain_updates: bool,
) -> PrefillLoopOutput:
    tokens, hidden_size = _validate_common(
        layers,
        hidden_states,
        position_embeddings,
        attention_mask,
    )
    if active_mask.shape != (1, tokens) or active_mask.dtype != torch.bool:
        raise ValueError("active_mask must be bool [1, tokens]")
    positions = torch.nonzero(active_mask[0], as_tuple=False).flatten()
    if positions.numel() == 0:
        raise ValueError("at least one prompt row must remain active")
    if composite_cache is not None and len(composite_cache.keys) != len(layers):
        raise ValueError("composite cache layer count differs from the model")

    keys = list(composite_cache.keys) if composite_cache is not None else []
    values = list(composite_cache.values) if composite_cache is not None else []
    updates: list[PrefillLayerUpdate] = []
    hidden = hidden_states
    cos = position_embeddings[0].index_select(1, positions)
    sin = position_embeddings[1].index_select(1, positions)

    for layer_index, layer in enumerate(layers):
        active_hidden = hidden.index_select(1, positions)
        residual = active_hidden
        normalized = layer.input_layernorm(active_hidden)
        attention = layer.self_attn
        head_dim = int(attention.head_dim)
        query_heads = int(attention.q_proj.out_features // head_dim)
        kv_heads = int(attention.k_proj.out_features // head_dim)
        active_rows = int(positions.numel())
        query_raw, key_raw, value_raw = project_qkv(attention, normalized)
        query = query_raw.view(
            1, active_rows, query_heads, head_dim
        ).transpose(1, 2)
        key = key_raw.view(
            1, active_rows, kv_heads, head_dim
        ).transpose(1, 2)
        value = value_raw.view(
            1, active_rows, kv_heads, head_dim
        ).transpose(1, 2)
        query, key = _apply_rope(query, key, cos, sin)

        if composite_cache is None:
            if active_rows != tokens:
                raise ValueError("the first dense composite cache must cover every token")
            full_key = key
            full_value = value
            keys.append(full_key)
            values.append(full_value)
        else:
            previous_key = composite_cache.keys[layer_index]
            previous_value = composite_cache.values[layer_index]
            expected = (1, kv_heads, tokens, head_dim)
            if tuple(previous_key.shape) != expected or previous_value.shape != previous_key.shape:
                raise ValueError("composite cache shape differs from the attention projection")
            full_key = previous_key if consume_inputs else previous_key.clone()
            full_value = previous_value if consume_inputs else previous_value.clone()
            full_key.index_copy_(2, positions, key)
            full_value.index_copy_(2, positions, value)
            keys[layer_index] = full_key
            values[layer_index] = full_value

        repeated_key = _repeat_kv(full_key, int(attention.num_key_value_groups))
        repeated_value = _repeat_kv(full_value, int(attention.num_key_value_groups))
        dense_rows = active_rows == tokens
        if dense_rows:
            selected_mask = None
            is_causal = True
        else:
            if attention_mask is None:
                key_positions = torch.arange(tokens, device=hidden.device)
                allowed = key_positions[None, :] <= positions[:, None]
                selected_mask = torch.zeros(
                    (1, 1, active_rows, tokens),
                    device=hidden.device,
                    dtype=query.dtype,
                ).masked_fill(~allowed[None, None], float("-inf"))
            else:
                selected_mask = attention_mask.index_select(-2, positions).to(query.dtype)
            is_causal = False
        attention_output = F.scaled_dot_product_attention(
            query,
            repeated_key,
            repeated_value,
            attn_mask=selected_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )
        attention_output = attention_output.transpose(1, 2).reshape(
            1, active_rows, hidden_size
        )
        attention_output = attention.o_proj(attention_output)
        active_hidden = residual + layer.input_layernorm_2(attention_output)

        residual = active_hidden
        mlp_output = fused_mlp(
            layer.mlp,
            layer.post_attention_layernorm(active_hidden),
        )
        active_hidden = residual + layer.post_attention_layernorm_2(mlp_output)
        if consume_inputs:
            hidden.index_copy_(1, positions, active_hidden)
        else:
            next_hidden = hidden.clone()
            next_hidden.index_copy_(1, positions, active_hidden)
            hidden = next_hidden
        if retain_updates:
            updates.append(
                PrefillLayerUpdate(
                    layer_index=layer_index,
                    positions=positions.to(torch.int32),
                    keys=key,
                    values=value,
                )
            )

    normalized_active = final_norm(hidden.index_select(1, positions))
    output_hidden = hidden if consume_inputs else hidden.clone()
    output_hidden.index_copy_(1, positions, normalized_active)
    active_rows = int(positions.numel())
    layer_count = len(layers)
    return PrefillLoopOutput(
        hidden_states=output_hidden,
        composite_cache=PrefillCompositeCache(keys=keys, values=values),
        updates=tuple(updates),
        receipt=PrefillReceipt(
            loop=int(loop),
            total_rows=tokens,
            active_rows=active_rows,
            qkv_rows=3 * active_rows * layer_count,
            output_projection_rows=active_rows * layer_count,
            mlp_rows=active_rows * layer_count,
            dense_target_calls=layer_count if not sparse else 0,
            sparse_target_calls=layer_count if sparse else 0,
            original_positions=tuple(int(value) for value in positions.tolist()),
        ),
    )


def dense_prefill_loop(
    layers: Sequence[nn.Module],
    hidden_states: torch.Tensor,
    *,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    final_norm: nn.Module,
    loop: int,
    composite_cache: PrefillCompositeCache | None = None,
) -> PrefillLoopOutput:
    """Execute one dense recurrent loop and replace every composite K/V row."""

    active = torch.ones(
        hidden_states.shape[:2],
        dtype=torch.bool,
        device=hidden_states.device,
    )
    return _execute_prefill_loop(
        layers,
        hidden_states,
        active_mask=active,
        composite_cache=composite_cache,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        final_norm=final_norm,
        loop=loop,
        sparse=False,
        consume_inputs=False,
        retain_updates=True,
    )


def sparse_prefill_loop(
    layers: Sequence[nn.Module],
    hidden_states: torch.Tensor,
    *,
    active_mask: torch.Tensor,
    composite_cache: PrefillCompositeCache,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    final_norm: nn.Module,
    loop: int,
    eligible_mask: torch.Tensor | None = None,
    consume_inputs: bool = False,
    retain_updates: bool = True,
) -> PrefillLoopOutput:
    """Execute a late Ouro loop only for active prompt rows."""

    if int(loop) not in (3, 4):
        raise ValueError("sparse prefill is defined for Loop 3 or Loop 4")
    if eligible_mask is not None:
        if eligible_mask.shape != active_mask.shape or eligible_mask.dtype != torch.bool:
            raise ValueError("eligible_mask must be bool and match active_mask")
        if not bool(torch.all(~active_mask | eligible_mask).item()):
            raise ValueError("Loop-4 active rows must be nested in Loop-3 rows")
    return _execute_prefill_loop(
        layers,
        hidden_states,
        active_mask=active_mask,
        composite_cache=composite_cache,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        final_norm=final_norm,
        loop=loop,
        sparse=True,
        consume_inputs=bool(consume_inputs),
        retain_updates=bool(retain_updates),
    )
