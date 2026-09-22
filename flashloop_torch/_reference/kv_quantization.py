#!/usr/bin/env python3
"""Reference accounting and fake quantization for cross-loop KV innovations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .kv_audit import validated_cross_loop_kv_audit


CODEC_PROFILES: dict[str, dict[str, Any]] = {
    "legacy": {
        "key_axes": ("per_channel",) * 4,
        "value_axes": ("per_token",) * 4,
        "key_bits": (2, 2, 2, 2),
        "value_bits": (4, 4, 2, 2),
    },
    "kivi_channel": {
        "key_axes": ("per_channel",) * 4,
        "value_axes": ("per_token",) * 4,
        "key_bits": (2, 2, 2, 2),
        "value_bits": (2, 2, 2, 2),
    },
    "kivi2_channel": {
        "key_axes": ("per_channel",) * 4,
        "value_axes": ("per_token",) * 4,
        "key_bits": (2, 2, 2, 2),
        "value_bits": (2, 2, 2, 2),
    },
    "kivi_token_delta": {
        "key_axes": ("per_channel", "per_token", "per_token", "per_token"),
        "value_axes": ("per_token",) * 4,
        "key_bits": (2, 2, 2, 2),
        "value_bits": (2, 2, 2, 2),
    },
    "kivi4_channel": {
        "key_axes": ("per_channel",) * 4,
        "value_axes": ("per_token",) * 4,
        "key_bits": (4, 4, 4, 4),
        "value_bits": (4, 4, 4, 4),
    },
    "kivi4_token_delta": {
        "key_axes": ("per_channel", "per_token", "per_token", "per_token"),
        "value_axes": ("per_token",) * 4,
        "key_bits": (4, 4, 4, 4),
        "value_bits": (4, 4, 4, 4),
    },
}


def codec_profile_for_method(method: str) -> str:
    if method == "flashloop_quantized_kv":
        return "legacy"
    if method == "flashloop_kivi4_channel_delta":
        return "kivi4_channel"
    if method == "flashloop_kivi2_channel_delta":
        return "kivi2_channel"
    if method == "cross_loop_kivi_channel_delta":
        return "kivi_channel"
    if method == "cross_loop_kivi_token_delta":
        return "kivi_token_delta"
    if method == "cross_loop_kivi4_channel_delta":
        return "kivi4_channel"
    if method == "cross_loop_kivi4_token_delta":
        return "kivi4_token_delta"
    raise ValueError(f"method {method} does not select a cross-loop KV profile")


def cross_loop_kv_metadata(method: str) -> dict[str, Any]:
    from .method_utils import method_uses_quantized_cross_loop_kv

    enabled = method_uses_quantized_cross_loop_kv(method)
    disabled = {
        "quantized_cross_loop_kv": enabled,
        "kv_quantization_reference": "fake_quant_materialized" if enabled else None,
        "kv_key_domain": "post_rope" if enabled else None,
        "kv_key_bits": [2, 2, 2, 2] if enabled else None,
        "kv_value_bits": [4, 4, 2, 2] if enabled else None,
        "kv_key_axis": "per_channel_over_tokens" if enabled else None,
        "kv_value_axis": "per_token_over_channels" if enabled else None,
        "kv_group_size": 64 if enabled else None,
        "kv_residual_length": 64 if enabled else None,
        "kv_quantization_scheme": "asymmetric_minmax" if enabled else None,
    }
    if not enabled or method == "flashloop_quantized_kv":
        return disabled
    profile_name = codec_profile_for_method(method)
    profile = CODEC_PROFILES[profile_name]
    key_axes = list(profile["key_axes"])
    return {
        **disabled,
        "kv_codec_profile": profile_name,
        "kv_key_bits": list(profile["key_bits"]),
        "kv_value_bits": list(profile["value_bits"]),
        "kv_key_axis": (
            "per_channel_over_tokens"
            if len(set(key_axes)) == 1
            else "loop1_per_channel_deltas_per_token"
        ),
        "kv_base_key_axis": "per_channel_over_tokens",
        "kv_delta_key_axis": (
            "per_channel_over_tokens"
            if key_axes[1] == "per_channel"
            else "per_token_over_channels"
        ),
        "kv_value_axis": "per_token_over_channels",
    }


@dataclass(frozen=True)
class TemporalGroupResult:
    reconstructed_keys: tuple[torch.Tensor, ...]
    reconstructed_values: tuple[torch.Tensor, ...]
    audit: dict[str, Any]


class CrossLoopKVCacheCompressor:
    """Streaming fake-quantizer for a one-to-four-loop Ouro cache prefix.

    Complete token groups leaving the BF16 residual tail are encoded once.
    Cache tensors remain materialized dequantized references so this class
    measures algorithm quality, not physical peak-memory reduction.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        recurrent_steps: int = 4,
        group_size: int = 64,
        residual_length: int = 64,
        codec_profile: str = "legacy",
    ) -> None:
        self.num_layers = int(num_layers)
        self.recurrent_steps = int(recurrent_steps)
        self.group_size = int(group_size)
        self.residual_length = int(residual_length)
        if codec_profile not in CODEC_PROFILES:
            raise ValueError(f"unknown codec profile {codec_profile}")
        self.codec_profile = codec_profile
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 1 <= self.recurrent_steps <= 4:
            raise ValueError("recurrent_steps must be in [1, 4]")
        if self.group_size <= 0 or self.residual_length < 0:
            raise ValueError("invalid group_size or residual_length")
        self.next_start = [0 for _ in range(self.num_layers)]
        self.quantized_groups = 0
        self.packed_stream_bytes = 0
        self.group_audits: list[dict[str, Any]] = []
        self.prefill_masks: tuple[torch.Tensor, ...] | None = None
        self.max_total_tokens = 0

    def set_prefill_masks(
        self,
        mask_loop3: torch.Tensor,
        mask_loop4: torch.Tensor,
    ) -> None:
        if mask_loop3.shape != mask_loop4.shape or mask_loop3.ndim != 2:
            raise ValueError("prefill masks must share shape [batch, tokens]")
        mask_loop3 = mask_loop3.detach().bool()
        mask_loop4 = mask_loop4.detach().bool()
        if not torch.all(~mask_loop4 | mask_loop3).item():
            raise ValueError("Loop-4 prefill mask must be nested in Loop-3")
        self.prefill_masks = (mask_loop3, mask_loop4)

    def _transition_masks(
        self,
        *,
        batch: int,
        start: int,
        stop: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        width = stop - start
        masks = [
            torch.ones((batch, width), dtype=torch.bool, device=device)
            for _ in range(self.recurrent_steps - 1)
        ]
        if self.prefill_masks is not None:
            for transition_index, source in enumerate(
                self.prefill_masks, start=1
            ):
                if transition_index >= len(masks):
                    break
                destination = masks[transition_index]
                overlap_stop = min(stop, int(source.shape[1]))
                if overlap_stop > start:
                    destination[:, : overlap_stop - start] = source[
                        :, start:overlap_stop
                    ].to(device)
        return tuple(masks)

    def compress_available(self, cache: Any) -> dict[str, Any]:
        required_slots = self.recurrent_steps * self.num_layers
        if len(cache.key_cache) < required_slots or len(cache.value_cache) < required_slots:
            raise RuntimeError("cache does not contain every requested loop slot")
        for layer in range(self.num_layers):
            slots = [
                loop * self.num_layers + layer
                for loop in range(self.recurrent_steps)
            ]
            keys = [cache.key_cache[index] for index in slots]
            values = [cache.value_cache[index] for index in slots]
            if any(tensor is None for tensor in (*keys, *values)):
                raise RuntimeError("cache contains an empty loop slot")
            total_tokens = int(keys[0].shape[2])
            if any(int(tensor.shape[2]) != total_tokens for tensor in (*keys, *values)):
                raise RuntimeError("loop cache slots have inconsistent lengths")
            self.max_total_tokens = max(self.max_total_tokens, total_tokens)
            while total_tokens - self.next_start[layer] >= self.residual_length + self.group_size:
                start = self.next_start[layer]
                stop = start + self.group_size
                key_group = [tensor[:, :, start:stop, :] for tensor in keys]
                value_group = [tensor[:, :, start:stop, :] for tensor in values]
                masks = self._transition_masks(
                    batch=int(keys[0].shape[0]),
                    start=start,
                    stop=stop,
                    device=keys[0].device,
                )
                result = encode_temporal_group(
                    key_group,
                    value_group,
                    active_masks=masks,
                    group_size=self.group_size,
                    codec_profile=self.codec_profile,
                )
                for loop, slot in enumerate(slots):
                    cache.key_cache[slot][:, :, start:stop, :].copy_(
                        result.reconstructed_keys[loop]
                    )
                    cache.value_cache[slot][:, :, start:stop, :].copy_(
                        result.reconstructed_values[loop]
                    )
                self.next_start[layer] = stop
                self.quantized_groups += 1
                self.packed_stream_bytes += int(result.audit["packed_bytes"])
                self.group_audits.append(result.audit)
        return self.audit(cache)

    def audit(self, cache: Any) -> dict[str, Any]:
        first = cache.key_cache[0]
        if first is None:
            raise RuntimeError("cache is empty")
        current_tokens = int(first.shape[2])
        total_tokens = max(self.max_total_tokens, current_tokens)
        batch, heads, _tokens, channels = (int(value) for value in first.shape)
        unique_quantized_tokens = min(self.next_start) if self.next_start else 0
        tail_tokens = total_tokens - unique_quantized_tokens
        residual_bf16_bytes = (
            self.num_layers
            * self.recurrent_steps
            * 2
            * batch
            * heads
            * tail_tokens
            * channels
            * 2
        )
        dense_recurrent_bf16_bytes = (
            self.num_layers
            * self.recurrent_steps
            * 2
            * batch
            * heads
            * total_tokens
            * channels
            * 2
        )
        mask_bytes = 0
        if self.prefill_masks is not None:
            mask_bytes = math.ceil(2 * int(self.prefill_masks[0].numel()) / 8)
        total_packed_bytes = self.packed_stream_bytes + residual_bf16_bytes + mask_bytes
        return {
            "status": "PASS",
            "kind": "asymmetric_cross_loop_innovation_fake_quant",
            "codec_profile": self.codec_profile,
            "recurrent_steps": self.recurrent_steps,
            "key_axes": list(
                CODEC_PROFILES[self.codec_profile]["key_axes"][: self.recurrent_steps]
            ),
            "value_axes": list(
                CODEC_PROFILES[self.codec_profile]["value_axes"][: self.recurrent_steps]
            ),
            "k_bits": list(
                CODEC_PROFILES[self.codec_profile]["key_bits"][: self.recurrent_steps]
            ),
            "v_bits": list(
                CODEC_PROFILES[self.codec_profile]["value_bits"][: self.recurrent_steps]
            ),
            "group_size": self.group_size,
            "residual_length": self.residual_length,
            "quantized_groups": self.quantized_groups,
            "quantized_tokens": unique_quantized_tokens,
            "bf16_tail_tokens": tail_tokens,
            "total_cache_tokens": total_tokens,
            "current_cache_tokens": current_tokens,
            "packed_stream_bytes": self.packed_stream_bytes,
            "residual_bf16_bytes": residual_bf16_bytes,
            "mask_bytes": mask_bytes,
            "total_packed_bytes": total_packed_bytes,
            "dense_recurrent_bf16_bytes": dense_recurrent_bf16_bytes,
            "dense_r4_bf16_bytes": (
                dense_recurrent_bf16_bytes if self.recurrent_steps == 4 else None
            ),
            "dense_recurrent_ratio": (
                total_packed_bytes / dense_recurrent_bf16_bytes
                if dense_recurrent_bf16_bytes
                else None
            ),
            "dense_r4_ratio": (
                total_packed_bytes / dense_recurrent_bf16_bytes
                if self.recurrent_steps == 4 and dense_recurrent_bf16_bytes
                else None
            ),
            "reference_cache_materialized": True,
        }


class CrossLoopKVQuantizationIntervention:
    """Attach the streaming fake-quantizer to a model's completed forwards."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        prefill_intervention: Any | None = None,
        group_size: int = 64,
        residual_length: int = 64,
        method: str = "flashloop_quantized_kv",
    ) -> None:
        recurrent_steps = int(getattr(model.config, "total_ut_steps", 0))
        if not 1 <= recurrent_steps <= 4:
            raise ValueError("cross-loop KV codec requires one to four Ouro loops")
        self.prefill_intervention = prefill_intervention
        self.compressor = CrossLoopKVCacheCompressor(
            num_layers=int(model.config.num_hidden_layers),
            recurrent_steps=recurrent_steps,
            group_size=group_size,
            residual_length=residual_length,
            codec_profile=codec_profile_for_method(method),
        )
        self.forward_calls = 0
        self.cache_updates = 0
        self.last_cache: Any | None = None
        self.closed = False
        self.handle = model.register_forward_hook(self._after_forward, with_kwargs=True)

    def _capture_prefill_masks(self) -> None:
        if self.compressor.prefill_masks is not None or self.prefill_intervention is None:
            return
        active_masks = getattr(self.prefill_intervention, "active_masks", {})
        if 2 in active_masks and 3 in active_masks:
            self.compressor.set_prefill_masks(active_masks[2], active_masks[3])

    def _after_forward(self, _module, _args, _kwargs, output):
        self.forward_calls += 1
        cache = getattr(output, "past_key_values", None)
        if cache is None:
            return None
        self._capture_prefill_masks()
        self.compressor.compress_available(cache)
        self.last_cache = cache
        self.cache_updates += 1
        return None

    def audit(self) -> dict[str, Any]:
        if self.last_cache is None:
            codec = {
                "status": "PENDING",
                "quantized_groups": 0,
                "quantized_tokens": 0,
            }
        else:
            codec = self.compressor.audit(self.last_cache)
        return {
            **codec,
            "forward_calls": self.forward_calls,
            "cache_updates": self.cache_updates,
            "closed": self.closed,
        }

    def close(self) -> None:
        if self.closed:
            return
        self.handle.remove()
        self.closed = True


def _asymmetric_fake_quant(
    values: torch.Tensor,
    *,
    bits: int,
    reduce_dim: int,
) -> torch.Tensor:
    bits = int(bits)
    if bits not in (2, 4):
        raise ValueError("reference codec supports only 2-bit or 4-bit streams")
    if values.numel() == 0:
        return values.clone()
    work = values.float()
    minimum = work.amin(dim=reduce_dim, keepdim=True)
    maximum = work.amax(dim=reduce_dim, keepdim=True)
    levels = float((1 << bits) - 1)
    span = maximum - minimum
    scale = torch.where(span > 0, span / levels, torch.ones_like(span))
    codes = torch.round((work - minimum) / scale).clamp_(0, levels)
    reconstructed = codes * scale + minimum
    return reconstructed.to(values.dtype)


def quantize_dequantize_k_per_channel(
    values: torch.Tensor,
    *,
    bits: int,
    group_size: int = 64,
) -> torch.Tensor:
    """Fake-quantize K independently per channel over token groups."""
    if values.ndim != 4:
        raise ValueError("K must have shape [batch, heads, tokens, channels]")
    group_size = int(group_size)
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    chunks = [
        _asymmetric_fake_quant(values[:, :, start : start + group_size, :], bits=bits, reduce_dim=2)
        for start in range(0, values.shape[2], group_size)
    ]
    return torch.cat(chunks, dim=2) if chunks else values.clone()


def quantize_dequantize_v_per_token(
    values: torch.Tensor,
    *,
    bits: int,
    group_size: int = 64,
) -> torch.Tensor:
    """Fake-quantize V independently per token over channel groups."""
    if values.ndim != 4:
        raise ValueError("V must have shape [batch, heads, tokens, channels]")
    group_size = int(group_size)
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    chunks = [
        _asymmetric_fake_quant(values[..., start : start + group_size], bits=bits, reduce_dim=3)
        for start in range(0, values.shape[3], group_size)
    ]
    return torch.cat(chunks, dim=3) if chunks else values.clone()


def _quantize_active_delta(
    delta: torch.Tensor,
    mask: torch.Tensor,
    *,
    bits: int,
    axis: str,
    group_size: int,
) -> torch.Tensor:
    if mask.shape != (delta.shape[0], delta.shape[2]):
        raise ValueError("transition mask must have shape [batch, tokens]")
    output = torch.zeros_like(delta)
    for batch_index in range(delta.shape[0]):
        positions = torch.nonzero(mask[batch_index], as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        selected = delta[batch_index : batch_index + 1].index_select(
            2, positions.to(delta.device)
        )
        if axis == "per_channel":
            selected = quantize_dequantize_k_per_channel(
                selected, bits=bits, group_size=group_size
            )
        elif axis == "per_token":
            selected = quantize_dequantize_v_per_token(
                selected, bits=bits, group_size=group_size
            )
        else:
            raise ValueError(f"unknown quantization axis {axis}")
        output[batch_index : batch_index + 1].index_copy_(
            2, positions.to(output.device), selected
        )
    return output


def _payload_bytes(values: int, bits: int) -> int:
    return math.ceil(int(values) * int(bits) / 8)


def _k_stream_bytes(shape: torch.Size, active_values: int, bits: int) -> int:
    batch, heads, _tokens, channels = (int(value) for value in shape)
    if active_values == 0:
        return 0
    active_tokens = active_values // (heads * channels)
    groups = batch * heads * channels if active_tokens else 0
    return _payload_bytes(active_values, bits) + groups * 4


def _v_stream_bytes(
    shape: torch.Size,
    active_values: int,
    bits: int,
    group_size: int,
) -> int:
    batch, heads, _tokens, channels = (int(value) for value in shape)
    if active_values == 0:
        return 0
    active_tokens = active_values // (heads * channels)
    channel_groups = math.ceil(channels / int(group_size))
    groups = active_tokens * heads * channel_groups
    return _payload_bytes(active_values, bits) + groups * 4


def _stream_bytes(
    shape: torch.Size,
    active_values: int,
    bits: int,
    axis: str,
    group_size: int,
) -> int:
    if axis == "per_channel":
        return _k_stream_bytes(shape, active_values, bits)
    if axis == "per_token":
        return _v_stream_bytes(shape, active_values, bits, group_size)
    raise ValueError(f"unknown quantization axis {axis}")


def encode_temporal_group(
    keys: Sequence[torch.Tensor],
    values: Sequence[torch.Tensor],
    *,
    active_masks: Sequence[torch.Tensor] | None = None,
    group_size: int = 64,
    codec_profile: str = "legacy",
) -> TemporalGroupResult:
    recurrent_steps = len(keys)
    if recurrent_steps != len(values) or not 1 <= recurrent_steps <= 4:
        raise ValueError("one to four equally sized loop K/V tensors are required")
    shape = keys[0].shape
    if len(shape) != 4 or any(tensor.shape != shape for tensor in (*keys, *values)):
        raise ValueError("all K/V tensors must share [batch, heads, tokens, channels]")
    if active_masks is None:
        batch, _heads, tokens, _channels = shape
        active_masks = tuple(
            torch.ones((batch, tokens), dtype=torch.bool, device=keys[0].device)
            for _ in range(recurrent_steps - 1)
        )
    if len(active_masks) != recurrent_steps - 1:
        raise ValueError("one transition mask is required per adjacent-loop pair")
    active_masks = tuple(mask.to(device=keys[0].device, dtype=torch.bool) for mask in active_masks)

    if codec_profile not in CODEC_PROFILES:
        raise ValueError(f"unknown codec profile {codec_profile}")
    profile = CODEC_PROFILES[codec_profile]
    key_axes = tuple(profile["key_axes"][:recurrent_steps])
    value_axes = tuple(profile["value_axes"][:recurrent_steps])
    key_bits = tuple(int(value) for value in profile["key_bits"][:recurrent_steps])
    value_bits = tuple(int(value) for value in profile["value_bits"][:recurrent_steps])

    def quantize(values: torch.Tensor, *, bits: int, axis: str) -> torch.Tensor:
        if axis == "per_channel":
            return quantize_dequantize_k_per_channel(
                values, bits=bits, group_size=group_size
            )
        if axis == "per_token":
            return quantize_dequantize_v_per_token(
                values, bits=bits, group_size=group_size
            )
        raise ValueError(f"unknown quantization axis {axis}")

    reconstructed_keys: list[torch.Tensor] = [
        quantize(keys[0], bits=key_bits[0], axis=key_axes[0])
    ]
    reconstructed_values: list[torch.Tensor] = [
        quantize(values[0], bits=value_bits[0], axis=value_axes[0])
    ]
    for transition_index, mask in enumerate(active_masks, start=1):
        key_delta = keys[transition_index] - reconstructed_keys[-1]
        value_delta = values[transition_index] - reconstructed_values[-1]
        reconstructed_keys.append(
            reconstructed_keys[-1]
            + _quantize_active_delta(
                key_delta,
                mask,
                bits=key_bits[transition_index],
                axis=key_axes[transition_index],
                group_size=group_size,
            )
        )
        reconstructed_values.append(
            reconstructed_values[-1]
            + _quantize_active_delta(
                value_delta,
                mask,
                bits=value_bits[transition_index],
                axis=value_axes[transition_index],
                group_size=group_size,
            )
        )

    batch, heads, tokens, channels = (int(value) for value in shape)
    full_values = batch * heads * tokens * channels
    transition_values = [
        int(mask.sum().item()) * heads * channels for mask in active_masks
    ]
    k_counts = [full_values, *transition_values]
    v_counts = [full_values, *transition_values]
    k_bits = list(key_bits)
    v_bits = list(value_bits)
    k_bytes = [
        _stream_bytes(shape, count, bits, axis, group_size)
        for count, bits, axis in zip(k_counts, k_bits, key_axes, strict=True)
    ]
    v_bytes = [
        _stream_bytes(shape, count, bits, axis, group_size)
        for count, bits, axis in zip(v_counts, v_bits, value_axes, strict=True)
    ]
    packed_bytes = sum(k_bytes) + sum(v_bytes)
    dense_recurrent_bf16_bytes = recurrent_steps * 2 * full_values * 2
    single_loop_bf16_bytes = 2 * full_values * 2
    audit = {
        "packed_bytes": packed_bytes,
        "recurrent_steps": recurrent_steps,
        "dense_recurrent_bf16_bytes": dense_recurrent_bf16_bytes,
        "dense_r4_bf16_bytes": (
            dense_recurrent_bf16_bytes if recurrent_steps == 4 else None
        ),
        "dense_recurrent_ratio": packed_bytes / dense_recurrent_bf16_bytes,
        "dense_r4_ratio": (
            packed_bytes / dense_recurrent_bf16_bytes
            if recurrent_steps == 4
            else None
        ),
        "single_loop_copy_ratio": packed_bytes / single_loop_bf16_bytes,
        "k_stream_bytes": k_bytes,
        "v_stream_bytes": v_bytes,
        "active_transition_tokens": [int(mask.sum().item()) for mask in active_masks],
        "closed_loop_prediction": True,
        "codec_profile": codec_profile,
        "key_axes": list(key_axes),
        "value_axes": list(value_axes),
        "k_bits": k_bits,
        "v_bits": v_bits,
    }
    return TemporalGroupResult(
        tuple(reconstructed_keys), tuple(reconstructed_values), audit
    )
