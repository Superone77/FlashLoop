"""Batch-one Ouro inference orchestration for FlashLoop."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch

from .cache import CrossLoopKVCache, CrossLoopKVLayerBuilder
from .bf16_cache import BF16CrossLoopKVLayerBuilder
from .config import FlashLoopConfig, validate_ouro_model
from .prefill import PrefillReceipt, dense_prefill_loop, sparse_prefill_loop
from .selection import select_prefill_tokens
from .weights import fused_mlp, project_qkv, project_value, repack_projection_weights


@dataclass(frozen=True)
class EngineState:
    cache: CrossLoopKVCache
    next_position: int
    mask_loop3: torch.Tensor
    mask_loop4: torch.Tensor
    prefill_receipts: tuple[PrefillReceipt, ...]


@dataclass(frozen=True)
class PrefillOutput:
    logits: torch.Tensor
    state: EngineState


@dataclass(frozen=True)
class DecodeReceipt:
    dense_attention_calls: int
    sparse_attention_calls: int
    dense_late_loop_calls: int
    selected_keys_per_layer: tuple[int, ...]


@dataclass(frozen=True)
class DecodeOutput:
    logits: torch.Tensor
    state: EngineState
    receipt: DecodeReceipt


@dataclass(frozen=True)
class _LayerSource:
    projected_output: torch.Tensor
    indices: torch.Tensor
    valid: torch.Tensor
    global_mass: torch.Tensor
    selected_weights: torch.Tensor
    selected_output: torch.Tensor


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


def _hidden_delta_score(current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    numerator = torch.linalg.vector_norm(current.float() - previous.float(), dim=-1)
    denominator = torch.linalg.vector_norm(current.float(), dim=-1).clamp_min(1e-12)
    return numerator / denominator


def _pad_decode_indices(
    indices: torch.Tensor,
    *,
    block_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad decode indices to a stable capacity and return their validity mask.

    One spare block is reserved at exact boundaries so an autoregressive append
    does not immediately change the tensor shape and trigger a new GPU kernel
    specialization.
    """

    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    count = int(indices.shape[-1])
    capacity = (count // block_size + 1) * block_size
    padded = torch.zeros(
        (*indices.shape[:-1], capacity),
        dtype=indices.dtype,
        device=indices.device,
    )
    padded[..., :count] = indices
    valid = torch.zeros_like(padded, dtype=torch.bool)
    valid[..., :count] = True
    return padded, valid


class FlashLoopEngine:
    """Narrow adapter that reuses official Ouro weights without copying them."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        config: FlashLoopConfig | None = None,
        validate_model: bool = True,
    ) -> None:
        if validate_model:
            validate_ouro_model(model)
        elif int(getattr(model.config, "total_ut_steps", 0)) != 4:
            raise ValueError("FlashLoop requires exactly four recurrent loops")
        self.model = model.eval()
        self.config = config or FlashLoopConfig()
        self.layers = self.model.model.layers
        self.backbone = self.model.model
        if validate_model and self.config.fuse_projection_weights:
            repack_projection_weights(self.layers)
        self._state: EngineState | None = None

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        config: FlashLoopConfig | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
    ) -> "FlashLoopEngine":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        ).to(device)
        return cls(model, config=config, validate_model=True)

    @property
    def state(self) -> EngineState:
        if self._state is None:
            raise RuntimeError("engine cache is not initialized")
        return self._state

    def reset(self) -> None:
        self._state = None

    @torch.inference_mode()
    def prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        stage_observer: Callable[[str], None] | None = None,
    ) -> PrefillOutput:
        def stage(name: str) -> None:
            if stage_observer is not None:
                stage_observer(name)

        stage("start")
        if self._state is not None:
            raise RuntimeError("engine cache is already initialized; call reset()")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("the first FlashLoop engine supports batch size one")
        if input_ids.shape[1] <= 0:
            raise ValueError("prefill requires at least one prompt token")
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape or not bool(attention_mask.bool().all().item()):
                raise ValueError("padding is not supported by the first batch-one engine")

        hidden = self.backbone.embed_tokens(input_ids)
        tokens = int(input_ids.shape[1])
        positions = torch.arange(tokens, device=input_ids.device).unsqueeze(0)
        rope = self.backbone.rotary_emb(hidden, positions)
        # Dense SDPA uses ``is_causal=True`` and sparse prefill constructs only
        # its active-query mask, so retaining an [tokens, tokens] FP32 mask is
        # unnecessary (256 MiB at an 8K prompt).
        causal = None
        stage("inputs_ready")

        loop1 = dense_prefill_loop(
            self.layers,
            hidden,
            position_embeddings=rope,
            attention_mask=causal,
            final_norm=self.backbone.norm,
            loop=1,
        )
        hidden1 = loop1.hidden_states
        cache1 = loop1.composite_cache
        updates1: list[Any] = list(loop1.updates)
        receipt1 = loop1.receipt
        del loop1
        stage("loop1_dense")
        builders: list[CrossLoopKVLayerBuilder | BF16CrossLoopKVLayerBuilder] = []
        builder_type = (
            CrossLoopKVLayerBuilder
            if self.config.quantize_cross_loop_kv
            else BF16CrossLoopKVLayerBuilder
        )
        for layer_index in range(len(self.layers)):
            update1 = updates1[layer_index]
            builders.append(
                builder_type.from_loop1(
                    update1.keys,
                    update1.values,
                    group_size=self.config.group_size,
                    residual_length=self.config.residual_length,
                    absolute_late_overrides=self.config.absolute_late_overrides,
                    absolute_override_dtype=self.config.absolute_override_dtype,
                    reader_backend=self.config.kv_reader_backend,
                )
            )
            updates1[layer_index] = None
        del updates1, cache1
        stage("loop1_packed")
        loop2 = dense_prefill_loop(
            self.layers,
            hidden1,
            position_embeddings=rope,
            attention_mask=causal,
            final_norm=self.backbone.norm,
            loop=2,
        )
        hidden2 = loop2.hidden_states
        cache2 = loop2.composite_cache
        updates2: list[Any] = list(loop2.updates)
        receipt2 = loop2.receipt
        del loop2
        stage("loop2_dense")
        for layer_index, builder in enumerate(builders):
            update2 = updates2[layer_index]
            builder.add_loop2(update2.keys, update2.values)
        stage("loop2_delta_packed")
        fraction3, fraction4 = self.config.effective_prefill_fractions
        mask3 = select_prefill_tokens(
            _hidden_delta_score(hidden2, hidden1),
            fraction=fraction3,
        )
        del hidden1
        loop3 = sparse_prefill_loop(
            self.layers,
            hidden2,
            active_mask=mask3,
            composite_cache=cache2,
            position_embeddings=rope,
            attention_mask=causal,
            final_norm=self.backbone.norm,
            loop=3,
            consume_inputs=True,
            retain_updates=False,
        )
        hidden3 = loop3.hidden_states
        cache3 = loop3.composite_cache
        receipt3 = loop3.receipt
        del loop3, cache2
        stage("loop3_sparse")
        mask4 = select_prefill_tokens(
            _hidden_delta_score(hidden3, hidden2),
            fraction=fraction4,
            eligible=mask3,
        )
        positions3 = torch.nonzero(mask3[0], as_tuple=False).flatten().to(torch.int32)
        positions4 = torch.nonzero(mask4[0], as_tuple=False).flatten().to(torch.int32)
        positions3_long = positions3.long()
        positions4_long = positions4.long()
        ranks4_in3 = torch.searchsorted(positions3, positions4).long()
        loop3_at4_keys: list[torch.Tensor | None] = []
        loop3_at4_values: list[torch.Tensor | None] = []
        for key, value in zip(cache3.keys, cache3.values, strict=True):
            loop3_at4_keys.append(key.index_select(2, positions4_long))
            loop3_at4_values.append(value.index_select(2, positions4_long))
        del hidden2
        loop4 = sparse_prefill_loop(
            self.layers,
            hidden3,
            active_mask=mask4,
            eligible_mask=mask3,
            composite_cache=cache3,
            position_embeddings=rope,
            attention_mask=causal,
            final_norm=self.backbone.norm,
            loop=4,
            consume_inputs=True,
            retain_updates=False,
        )
        hidden4 = loop4.hidden_states
        cache4 = loop4.composite_cache
        receipt4 = loop4.receipt
        del loop4, cache3, hidden3
        stage("loop4_sparse")

        physical = CrossLoopKVCache(num_layers=len(self.layers))
        for layer_index in range(len(self.layers)):
            loop3_key = cache4.keys[layer_index].index_select(2, positions3_long)
            loop3_value = cache4.values[layer_index].index_select(2, positions3_long)
            if positions4.numel():
                snapshot_key = loop3_at4_keys[layer_index]
                snapshot_value = loop3_at4_values[layer_index]
                assert snapshot_key is not None and snapshot_value is not None
                loop3_key.index_copy_(2, ranks4_in3, snapshot_key)
                loop3_value.index_copy_(2, ranks4_in3, snapshot_value)
            loop4_key = cache4.keys[layer_index].index_select(2, positions4_long)
            loop4_value = cache4.values[layer_index].index_select(2, positions4_long)
            physical.set_layer(
                layer_index,
                builders[layer_index].finalize(
                    loop3_positions=positions3,
                    loop3_keys=loop3_key,
                    loop3_values=loop3_value,
                    loop4_positions=positions4,
                    loop4_keys=loop4_key,
                    loop4_values=loop4_value,
                ),
            )
            updates2[layer_index] = None
            loop3_at4_keys[layer_index] = None
            loop3_at4_values[layer_index] = None
            cache4.keys[layer_index] = torch.empty(0, device=hidden4.device)
            cache4.values[layer_index] = torch.empty(0, device=hidden4.device)
        stage("physical_cache_finalized")
        self._state = EngineState(
            cache=physical,
            next_position=tokens,
            mask_loop3=mask3,
            mask_loop4=mask4,
            prefill_receipts=(receipt1, receipt2, receipt3, receipt4),
        )
        logits = self.model.lm_head(hidden4[:, -1]).float()
        stage("logits_ready")
        return PrefillOutput(logits=logits, state=self._state)

    def _selected_value_aggregate(
        self,
        *,
        layer_index: int,
        loop: int,
        selected_indices: torch.Tensor,
        selected_weights: torch.Tensor,
        current_value: torch.Tensor,
        past_tokens: int,
    ) -> torch.Tensor:
        is_current = selected_indices == past_tokens
        safe = torch.where(is_current, torch.zeros_like(selected_indices), selected_indices)
        past_weights = selected_weights * (~is_current)[:, :, None, :]
        output = self.state.cache.pv(
            layer_index,
            loop,
            past_weights,
            safe,
            validate_indices=False,
        )
        output += torch.sum(
            selected_weights * is_current[:, :, None, :],
            dim=-1,
            keepdim=True,
        ) * current_value.float()
        return output

    @torch.inference_mode()
    def decode(self, token_ids: torch.Tensor) -> DecodeOutput:
        state = self.state
        if token_ids.shape != (1, 1):
            raise ValueError("decode requires exactly one token at batch size one")
        hidden = self.backbone.embed_tokens(token_ids)
        position_ids = torch.tensor(
            [[state.next_position]], device=token_ids.device, dtype=torch.long
        )
        rope = self.backbone.rotary_emb(hidden, position_ids)
        past_tokens = state.next_position
        current_keys: list[list[torch.Tensor | None]] = [
            [None] * 4 for _ in range(len(self.layers))
        ]
        current_values: list[list[torch.Tensor | None]] = [
            [None] * 4 for _ in range(len(self.layers))
        ]
        sources: list[_LayerSource | None] = [None] * len(self.layers)
        selected_counts: list[int] = []

        for loop in range(4):
            for layer_index, layer in enumerate(self.layers):
                residual = hidden
                normalized = layer.input_layernorm(hidden)
                attention = layer.self_attn
                head_dim = int(attention.head_dim)
                query_heads = int(attention.q_proj.out_features // head_dim)
                kv_heads = int(attention.k_proj.out_features // head_dim)
                if loop >= 2 and self.config.reuse_selected_probabilities:
                    value_raw = project_value(attention, normalized)
                    value = value_raw.view(
                        1, 1, kv_heads, head_dim
                    ).transpose(1, 2)
                    key = current_keys[layer_index][1]
                    assert key is not None
                    query = None
                else:
                    query_raw, key_raw, value_raw = project_qkv(attention, normalized)
                    query = query_raw.view(
                        1, 1, query_heads, head_dim
                    ).transpose(1, 2)
                    key = key_raw.view(1, 1, kv_heads, head_dim).transpose(1, 2)
                    value = value_raw.view(
                        1, 1, kv_heads, head_dim
                    ).transpose(1, 2)
                    query, key = _apply_rope(query, key, *rope)
                current_keys[layer_index][loop] = key
                current_values[layer_index][loop] = value

                if loop < 2 or not self.config.enable_sparse_decode:
                    assert query is not None
                    unprojected, source_logits = state.cache.dense_source_attention(
                        layer_index,
                        loop,
                        query=query,
                        current_key=key,
                        current_value=value,
                        scaling=float(attention.scaling),
                        return_logits=loop == 1 and self.config.enable_sparse_decode,
                    )
                    projected = attention.o_proj(
                        unprojected.to(query.dtype)
                        .transpose(1, 2)
                        .reshape(1, 1, -1)
                    )
                    if loop == 1 and self.config.enable_sparse_decode:
                        assert source_logits is not None
                        weights = torch.softmax(
                            source_logits, dim=-1, dtype=torch.float32
                        )
                        budget = min(
                            past_tokens + 1,
                            max(
                                1,
                                math.ceil(
                                    (past_tokens + 1)
                                    * self.config.effective_decode_key_fraction
                                ),
                            ),
                        )
                        top = torch.topk(weights, k=budget, dim=-1)
                        selected_positions = torch.sort(top.indices, dim=-1).values
                        logical_positions = torch.arange(
                            past_tokens + 1,
                            device=hidden.device,
                            dtype=torch.int32,
                        )[None, None].expand(1, query_heads, -1)
                        selected_indices = torch.gather(
                            logical_positions,
                            -1,
                            selected_positions.squeeze(-2).long(),
                        )
                        selected_global = torch.gather(
                            weights,
                            -1,
                            selected_positions,
                        )
                        selected_indices, selected_valid = _pad_decode_indices(
                            selected_indices
                        )
                        padded_global = torch.zeros(
                            (*selected_global.shape[:-1], selected_indices.shape[-1]),
                            device=selected_global.device,
                            dtype=selected_global.dtype,
                        )
                        padded_global[..., :budget] = selected_global
                        selected_global = padded_global
                        mass = selected_global.sum(dim=-1, keepdim=True)
                        selected_weights = selected_global / mass
                        selected_output = self._selected_value_aggregate(
                            layer_index=layer_index,
                            loop=1,
                            selected_indices=selected_indices,
                            selected_weights=selected_weights,
                            current_value=value,
                            past_tokens=past_tokens,
                        )
                        sources[layer_index] = _LayerSource(
                            projected_output=projected,
                            indices=selected_indices,
                            valid=selected_valid,
                            global_mass=mass,
                            selected_weights=selected_weights,
                            selected_output=selected_output,
                        )
                        selected_counts.append(budget)
                else:
                    source = sources[layer_index]
                    assert source is not None
                    if self.config.reuse_selected_probabilities:
                        target = self._selected_value_aggregate(
                            layer_index=layer_index,
                            loop=loop,
                            selected_indices=source.indices,
                            selected_weights=source.selected_weights,
                            current_value=value,
                            past_tokens=past_tokens,
                        )
                        delta = source.global_mass.float() * (
                            target - source.selected_output.float()
                        )
                    else:
                        assert query is not None
                        delta = state.cache.cached_mass_sparse_attention_delta(
                            layer_index,
                            loop,
                            query=query,
                            current_key=key,
                            current_value=value,
                            current_position=past_tokens,
                            indices=source.indices,
                            valid=source.valid,
                            source_selected_output=source.selected_output,
                            global_mass=source.global_mass,
                            scaling=float(attention.scaling),
                        )
                    projected = source.projected_output + attention.o_proj(
                        delta.to(value.dtype).transpose(1, 2).reshape(1, 1, -1)
                    )

                hidden = residual + layer.input_layernorm_2(projected)
                residual = hidden
                hidden = residual + layer.post_attention_layernorm_2(
                    fused_mlp(layer.mlp, layer.post_attention_layernorm(hidden))
                )
            hidden = self.backbone.norm(hidden)

        for layer_index, physical_layer in enumerate(state.cache.layers):
            assert physical_layer is not None
            layer_keys = current_keys[layer_index]
            layer_values = current_values[layer_index]
            assert all(value is not None for value in (*layer_keys, *layer_values))
            physical_layer.append_decode_token(layer_keys, layer_values)  # type: ignore[arg-type]

        self._state = EngineState(
            cache=state.cache,
            next_position=past_tokens + 1,
            mask_loop3=state.mask_loop3,
            mask_loop4=state.mask_loop4,
            prefill_receipts=state.prefill_receipts,
        )
        logits = self.model.lm_head(hidden[:, -1]).float()
        return DecodeOutput(
            logits=logits,
            state=self._state,
            receipt=DecodeReceipt(
                dense_attention_calls=(
                    (2 if self.config.enable_sparse_decode else 4) * len(self.layers)
                ),
                sparse_attention_calls=(
                    (2 if self.config.enable_sparse_decode else 0) * len(self.layers)
                ),
                dense_late_loop_calls=(
                    (0 if self.config.enable_sparse_decode else 2) * len(self.layers)
                ),
                selected_keys_per_layer=tuple(selected_counts),
            ),
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        stop_token_sequences: tuple[tuple[int, ...], ...] = (),
    ) -> torch.Tensor:
        if int(max_new_tokens) <= 0:
            raise ValueError("max_new_tokens must be positive")
        if eos_token_id is None:
            eos_token_id = getattr(self.model.config, "eos_token_id", None)
        prefill = self.prefill(input_ids)
        generated = input_ids
        logits = prefill.logits
        completion_tokens: list[int] = []
        for step in range(int(max_new_tokens)):
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            completion_tokens.append(int(next_token.item()))
            reached_eos = (
                eos_token_id is not None
                and completion_tokens[-1] == int(eos_token_id)
            )
            reached_suffix = any(
                sequence
                and len(completion_tokens) >= len(sequence)
                and tuple(completion_tokens[-len(sequence) :]) == sequence
                for sequence in stop_token_sequences
            )
            if reached_eos or reached_suffix:
                break
            if step + 1 < int(max_new_tokens):
                logits = self.decode(next_token).logits
        return generated
