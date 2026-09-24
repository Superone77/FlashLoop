"""Physical anchor-plus-delta Cross-Loop KV cache.

This module never stores a four-loop BF16 prompt cache once construction returns.
Its Torch gather path is a correctness reference; GPU serving uses the fused reader.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch

from .packing import (
    Packed4Bit,
    pack_k_per_channel,
    pack_v_per_token,
)
from .kivi_backend import (
    KiviPacked4Bit,
    kivi_cross_loop_pv,
    kivi_cross_loop_qk,
    kivi_fused_cross_loop_pv,
    kivi_fused_cross_loop_qk,
    kivi_fused_dense_source_attention,
    kivi_fused_sparse_attention_delta,
    kivi_outer_gemv,
    kivi_selected_pv,
    kivi_selected_qk,
    load_kivi_outer_extension,
    outer_matmul_reference,
    pack_kivi_k_per_channel,
    pack_kivi_v_per_token,
)


PackedKV = Packed4Bit | KiviPacked4Bit


_DENSE_SPLIT_CONTEXT_TOKENS = 2048


def _packed_attention_schedule(path: str, selected_rows: int) -> str:
    """Choose the measured A100 schedule for a packed attention reader.

    Dense readers cross over at 2K tokens because the fused online-softmax
    kernel serializes token tiles within one program per head.  Long dense
    reads use split-KV online softmax: independent token tiles followed by a
    stable merge, avoiding materialized logits and weights.  The default sparse
    sparse path remains fused: its isolated split-KV kernel is faster at 8K,
    but the extra system overhead regressed whole-model decode.
    """

    if path == "dense":
        return (
            "split_kv"
            if int(selected_rows) >= _DENSE_SPLIT_CONTEXT_TOKENS
            else "fused"
        )
    if path == "sparse":
        return "fused"
    raise ValueError(f"unknown packed attention path: {path}")


def _tensor_bytes(value: torch.Tensor) -> int:
    return int(value.numel() * value.element_size())


def _pack_key(
    values: torch.Tensor,
    group_size: int,
    reader_backend: str = "triton_row",
) -> PackedKV:
    if reader_backend == "kivi_outer":
        return pack_kivi_k_per_channel(values, group_size=group_size)
    if reader_backend != "triton_row":
        raise ValueError("unknown KV reader backend")
    if values.is_cuda and values.shape[-1] == 128 and int(group_size) == 64:
        from .kernels.quantization import triton_pack_4bit

        packed = triton_pack_4bit(values, axis="per_channel", group_size=group_size)
    else:
        packed = pack_k_per_channel(values, group_size=group_size)
    if packed.row_groups is not None:
        return packed
    row_groups = torch.div(
        torch.arange(
            packed.original_shape[2],
            device=packed.payload.device,
            dtype=torch.int32,
        ),
        int(group_size),
        rounding_mode="floor",
    )
    return Packed4Bit(
        payload=packed.payload,
        minimum=packed.minimum,
        scale=packed.scale,
        original_shape=packed.original_shape,
        original_dtype=packed.original_dtype,
        axis=packed.axis,
        group_size=packed.group_size,
        row_groups=row_groups,
    )


def _pack_value(
    values: torch.Tensor,
    group_size: int,
    reader_backend: str = "triton_row",
) -> PackedKV:
    if reader_backend == "kivi_outer":
        return pack_kivi_v_per_token(values, group_size=group_size)
    if reader_backend != "triton_row":
        raise ValueError("unknown KV reader backend")
    if values.is_cuda and values.shape[-1] == 128 and int(group_size) == 64:
        from .kernels.quantization import triton_pack_4bit

        return triton_pack_4bit(values, axis="per_token", group_size=group_size)
    return pack_v_per_token(values, group_size=group_size)


def _decode_packed(packed: PackedKV) -> torch.Tensor:
    """Decode a CUDA stream with one kernel instead of Torch tensor plumbing."""

    if isinstance(packed, Packed4Bit) and packed.payload.is_cuda:
        from .kernels.quantization import triton_gather_dequantize

        indices = torch.arange(
            packed.original_shape[2],
            device=packed.payload.device,
            dtype=torch.int32,
        )
        return triton_gather_dequantize(
            packed,
            indices,
            validate_indices=False,
        )
    return packed.decode()


def _decode_rows(packed: PackedKV, indices: torch.Tensor) -> torch.Tensor:
    """Reference-only logical row gather for audits and cache rollover."""

    if isinstance(packed, Packed4Bit):
        return packed.decode_rows(indices)
    return packed.decode().index_select(2, indices.long())


def _concat_packed(left: PackedKV | None, right: PackedKV | None) -> PackedKV | None:
    if left is None:
        return right
    if right is None:
        return left
    if isinstance(left, KiviPacked4Bit) or isinstance(right, KiviPacked4Bit):
        if not isinstance(left, KiviPacked4Bit) or not isinstance(right, KiviPacked4Bit):
            raise ValueError("cannot concatenate different packed KV layouts")
        if (
            left.axis != right.axis
            or left.group_size != right.group_size
            or left.original_dtype != right.original_dtype
            or left.original_shape[:2] != right.original_shape[:2]
            or left.original_shape[3] != right.original_shape[3]
        ):
            raise ValueError("cannot concatenate incompatible packed streams")
        left_tokens = int(left.original_shape[2])
        right_tokens = int(right.original_shape[2])
        # Decode rollover always appends a complete token group.  KIVI's K
        # payload is output-packed along tokens and its affine metadata is
        # grouped along tokens, so a group/pack-aligned boundary can be joined
        # directly.  V is independently quantized per token and is therefore
        # directly appendable at every token boundary.  Avoiding decode +
        # full-prefix repack makes rollover O(new_group) instead of O(context).
        directly_appendable = left.axis == "per_token" or (
            left_tokens % left.group_size == 0 and left_tokens % 8 == 0
        )
        if directly_appendable:
            token_dimension = 2 if left.axis == "per_channel" else 3
            return KiviPacked4Bit(
                payload=torch.cat(
                    (left.payload, right.payload), dim=token_dimension
                ).contiguous(),
                minimum=torch.cat(
                    (left.minimum, right.minimum), dim=token_dimension
                ).contiguous(),
                scale=torch.cat(
                    (left.scale, right.scale), dim=token_dimension
                ).contiguous(),
                original_shape=(
                    left.original_shape[0],
                    left.original_shape[1],
                    left_tokens + right_tokens,
                    left.original_shape[3],
                ),
                original_dtype=left.original_dtype,
                axis=left.axis,
                group_size=left.group_size,
            )
        merged = torch.cat((left.decode(), right.decode()), dim=2)
        if left.axis == "per_channel":
            return pack_kivi_k_per_channel(merged, group_size=left.group_size)
        return pack_kivi_v_per_token(merged, group_size=left.group_size)
    if (
        left.axis != right.axis
        or left.group_size != right.group_size
        or left.original_dtype != right.original_dtype
        or left.original_shape[:2] != right.original_shape[:2]
        or left.original_shape[3] != right.original_shape[3]
    ):
        raise ValueError("cannot concatenate incompatible packed streams")
    row_groups = None
    if left.axis == "per_channel":
        left_groups = left.row_groups
        if left_groups is None:
            left_groups = torch.div(
                torch.arange(left.original_shape[2], device=left.payload.device),
                left.group_size,
                rounding_mode="floor",
            ).to(torch.int32)
        right_groups = right.row_groups
        if right_groups is None:
            right_groups = torch.div(
                torch.arange(right.original_shape[2], device=right.payload.device),
                right.group_size,
                rounding_mode="floor",
            ).to(torch.int32)
        offset = int(left.minimum.shape[2])
        row_groups = torch.cat([left_groups, right_groups + offset], dim=0).contiguous()
    return Packed4Bit(
        payload=torch.cat([left.payload, right.payload], dim=2).contiguous(),
        minimum=torch.cat([left.minimum, right.minimum], dim=2).contiguous(),
        scale=torch.cat([left.scale, right.scale], dim=2).contiguous(),
        original_shape=(
            left.original_shape[0],
            left.original_shape[1],
            left.original_shape[2] + right.original_shape[2],
            left.original_shape[3],
        ),
        original_dtype=left.original_dtype,
        axis=left.axis,
        group_size=left.group_size,
        row_groups=row_groups,
    )


def _concat_optional_tensor(
    left: torch.Tensor | None,
    right: torch.Tensor | None,
) -> torch.Tensor | None:
    if left is None:
        return right
    if right is None:
        return left
    if (
        left.dtype != right.dtype
        or left.device != right.device
        or left.shape[:2] != right.shape[:2]
        or left.shape[3:] != right.shape[3:]
    ):
        raise ValueError("cannot concatenate incompatible absolute override streams")
    return torch.cat([left, right], dim=2).contiguous()


def _validate_loop_tensors(
    keys: Sequence[torch.Tensor],
    values: Sequence[torch.Tensor],
) -> tuple[int, int, int, int]:
    if len(keys) != 4 or len(values) != 4:
        raise ValueError("exactly four loop K/V tensors are required")
    shape = tuple(int(value) for value in keys[0].shape)
    if len(shape) != 4 or shape[0] != 1:
        raise ValueError("the first engine release requires [1, heads, tokens, channels]")
    if any(tuple(tensor.shape) != shape for tensor in (*keys, *values)):
        raise ValueError("all loop K/V tensors must have the same shape")
    if any(tensor.dtype != keys[0].dtype for tensor in (*keys, *values)):
        raise ValueError("all loop K/V tensors must have the same dtype")
    if any(tensor.device != keys[0].device for tensor in (*keys, *values)):
        raise ValueError("all loop K/V tensors must share a device")
    return shape


def _rank_map(tokens: int, positions: torch.Tensor, device: torch.device) -> torch.Tensor:
    result = torch.full((tokens,), -1, dtype=torch.int32, device=device)
    if positions.numel():
        result[positions.long()] = torch.arange(
            positions.numel(),
            dtype=torch.int32,
            device=device,
        )
    return result


def _normalize_logical_indices(
    indices: torch.Tensor,
    *,
    batch: int,
    heads: int,
    tokens: int,
    device: torch.device,
    validate_bounds: bool = True,
) -> torch.Tensor:
    if indices.ndim == 1:
        indices = indices[None, None, :].expand(batch, heads, -1)
    elif indices.ndim == 4 and indices.shape[-2] == 1:
        indices = indices.squeeze(-2)
    if indices.ndim != 3 or tuple(indices.shape[:2]) != (batch, heads):
        raise ValueError("indices must have shape [selected] or [batch, heads, selected]")
    normalized = indices.to(device=device, dtype=torch.int32).contiguous()
    if validate_bounds and normalized.numel() and (
        int(normalized.min().item()) < 0 or int(normalized.max().item()) >= tokens
    ):
        raise IndexError("cache index is outside the logical sequence")
    return normalized


class CrossLoopKVLayerBuilder:
    """Build one physical cache layer without retaining a dense Loop-1 prefix.

    The regular one-shot constructor is convenient for tests, but it forces the
    prefill path to keep the complete Loop-1 and Loop-2 BF16 caches alive at the
    same time.  This staged builder packs Loop 1 immediately, adds Loop 2 as a
    packed delta, and only then installs the compact Loop-3/4 overrides.
    """

    def __init__(self) -> None:
        self.group_size = 0
        self.residual_length = 0
        self.packed_tokens = 0
        self._shape: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._dtype = torch.float16
        self._device = torch.device("cpu")
        self._packed_keys: list[PackedKV | None] = [None] * 4
        self._packed_values: list[PackedKV | None] = [None] * 4
        self._absolute_keys: list[torch.Tensor | None] = [None] * 4
        self._absolute_values: list[torch.Tensor | None] = [None] * 4
        self._tail_keys: list[torch.Tensor] = []
        self._tail_values: list[torch.Tensor] = []
        self._has_loop2 = False
        self.absolute_late_overrides = False
        self.absolute_override_dtype = "int4"
        self.reader_backend = "triton_row"

    @classmethod
    def from_loop1(
        cls,
        loop1_key: torch.Tensor,
        loop1_value: torch.Tensor,
        *,
        group_size: int = 64,
        residual_length: int = 64,
        absolute_late_overrides: bool = False,
        absolute_override_dtype: str = "int4",
        reader_backend: str = "triton_row",
    ) -> "CrossLoopKVLayerBuilder":
        shape = tuple(int(value) for value in loop1_key.shape)
        if len(shape) != 4 or shape[0] != 1 or loop1_value.shape != loop1_key.shape:
            raise ValueError("Loop-1 K/V must match [1, heads, tokens, channels]")
        if (
            loop1_value.dtype != loop1_key.dtype
            or loop1_value.device != loop1_key.device
        ):
            raise ValueError("Loop-1 K/V must share dtype and device")
        group_size = int(group_size)
        residual_length = int(residual_length)
        if group_size <= 0 or residual_length < 0:
            raise ValueError("invalid group_size or residual_length")

        result = cls()
        result.group_size = group_size
        result.residual_length = residual_length
        result._shape = shape
        result._dtype = loop1_key.dtype
        result._device = loop1_key.device
        result.absolute_late_overrides = bool(absolute_late_overrides)
        if absolute_override_dtype not in {"int4", "bf16"}:
            raise ValueError("absolute_override_dtype must be 'int4' or 'bf16'")
        if absolute_override_dtype == "bf16" and not result.absolute_late_overrides:
            raise ValueError("bf16 absolute overrides require absolute_late_overrides=True")
        result.absolute_override_dtype = absolute_override_dtype
        if reader_backend not in {"kivi_outer", "triton_row"}:
            raise ValueError("unknown KV reader backend")
        if reader_backend == "kivi_outer" and result.absolute_late_overrides:
            raise ValueError("kivi_outer does not support absolute late overrides")
        result.reader_backend = reader_backend
        tokens = shape[2]
        result.packed_tokens = max(
            0, ((tokens - residual_length) // group_size) * group_size
        )
        if result.packed_tokens:
            result._packed_keys[0] = _pack_key(
                loop1_key[:, :, : result.packed_tokens, :], group_size, reader_backend
            )
            result._packed_values[0] = _pack_value(
                loop1_value[:, :, : result.packed_tokens, :], group_size, reader_backend
            )
        result._tail_keys = [loop1_key[:, :, result.packed_tokens :, :].clone()]
        result._tail_values = [loop1_value[:, :, result.packed_tokens :, :].clone()]
        return result

    @property
    def retains_dense_loop1_prefix(self) -> bool:
        """Audit flag used by the prefill memory regression test."""

        return False

    def _validate_dense_pair(
        self, key: torch.Tensor, value: torch.Tensor, *, label: str
    ) -> None:
        if tuple(key.shape) != self._shape or value.shape != key.shape:
            raise ValueError(f"{label} K/V shape differs from Loop 1")
        if key.dtype != self._dtype or value.dtype != self._dtype:
            raise ValueError(f"{label} K/V dtype differs from Loop 1")
        if key.device != self._device or value.device != self._device:
            raise ValueError(f"{label} K/V device differs from Loop 1")

    def add_loop2(self, loop2_key: torch.Tensor, loop2_value: torch.Tensor) -> None:
        if self._has_loop2:
            raise RuntimeError("Loop 2 is already installed")
        self._validate_dense_pair(loop2_key, loop2_value, label="Loop-2")
        if self.packed_tokens:
            anchor_key = self._packed_keys[0]
            anchor_value = self._packed_values[0]
            assert anchor_key is not None and anchor_value is not None
            reconstructed_key = _decode_packed(anchor_key)
            self._packed_keys[1] = _pack_key(
                loop2_key[:, :, : self.packed_tokens, :] - reconstructed_key,
                self.group_size,
                self.reader_backend,
            )
            del reconstructed_key
            reconstructed_value = _decode_packed(anchor_value)
            self._packed_values[1] = _pack_value(
                loop2_value[:, :, : self.packed_tokens, :] - reconstructed_value,
                self.group_size,
                self.reader_backend,
            )
            del reconstructed_value
        self._tail_keys.append(loop2_key[:, :, self.packed_tokens :, :].clone())
        self._tail_values.append(loop2_value[:, :, self.packed_tokens :, :].clone())
        self._has_loop2 = True

    def _validate_sparse_updates(
        self,
        positions: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        label: str,
    ) -> torch.Tensor:
        logical = positions.to(device=self._device, dtype=torch.int32).contiguous()
        if logical.ndim != 1 or (
            logical.numel() > 1
            and not bool(torch.all(logical[1:] > logical[:-1]).item())
        ):
            raise ValueError(f"{label} positions must be strictly sorted")
        tokens = self._shape[2]
        if logical.numel() and (
            int(logical.min().item()) < 0 or int(logical.max().item()) >= tokens
        ):
            raise IndexError(f"{label} update is outside the prompt")
        expected = (self._shape[0], self._shape[1], int(logical.numel()), self._shape[3])
        if tuple(key.shape) != expected or value.shape != key.shape:
            raise ValueError(f"{label} compact K/V shape differs from its positions")
        if key.dtype != self._dtype or value.dtype != self._dtype:
            raise ValueError(f"{label} compact K/V dtype differs from Loop 1")
        if key.device != self._device or value.device != self._device:
            raise ValueError(f"{label} compact K/V device differs from Loop 1")
        return logical

    def _install_sparse_streams(
        self,
        *,
        packed: list[PackedKV | None],
        absolute: list[torch.Tensor | None],
        targets: tuple[torch.Tensor, torch.Tensor],
        prefix_positions: tuple[torch.Tensor, torch.Tensor],
        prefix_update_ranks: tuple[torch.Tensor, torch.Tensor],
        pack: Callable[[torch.Tensor, int], PackedKV],
    ) -> None:
        if self.absolute_late_overrides:
            for stream_index in (2, 3):
                compact = prefix_update_ranks[stream_index - 2]
                if compact.numel():
                    target = targets[stream_index - 2].index_select(2, compact)
                    if self.absolute_override_dtype == "bf16":
                        absolute[stream_index] = target.to(torch.bfloat16).contiguous()
                    else:
                        packed[stream_index] = pack(target, self.group_size)
            return
        anchor = packed[0]
        delta2 = packed[1]
        assert anchor is not None and delta2 is not None
        reconstructed = _decode_packed(anchor) + _decode_packed(delta2)
        decoded3: torch.Tensor | None = None
        for stream_index in (2, 3):
            logical = prefix_positions[stream_index - 2]
            compact = prefix_update_ranks[stream_index - 2]
            if not logical.numel():
                continue
            target = targets[stream_index - 2].index_select(2, compact)
            base = reconstructed.index_select(2, logical.long())
            if stream_index == 3:
                assert decoded3 is not None
                ranks_in_loop3 = torch.searchsorted(prefix_positions[0], logical).long()
                base = base + decoded3.index_select(2, ranks_in_loop3)
            packed[stream_index] = pack(target - base, self.group_size)
            if stream_index == 2:
                assert packed[2] is not None
                decoded3 = _decode_packed(packed[2])

    def finalize(
        self,
        *,
        loop3_positions: torch.Tensor,
        loop3_keys: torch.Tensor,
        loop3_values: torch.Tensor,
        loop4_positions: torch.Tensor,
        loop4_keys: torch.Tensor,
        loop4_values: torch.Tensor,
    ) -> "CrossLoopKVLayer":
        if not self._has_loop2:
            raise RuntimeError("Loop 2 must be installed before finalization")
        positions3 = self._validate_sparse_updates(
            loop3_positions, loop3_keys, loop3_values, label="Loop-3"
        )
        positions4 = self._validate_sparse_updates(
            loop4_positions, loop4_keys, loop4_values, label="Loop-4"
        )
        if positions4.numel():
            ranks4 = torch.searchsorted(positions3, positions4)
            valid = ranks4 < positions3.numel()
            matched = torch.zeros_like(valid)
            matched[valid] = positions3[ranks4[valid]] == positions4[valid]
            if not bool(matched.all().item()):
                raise ValueError("Loop-4 positions must be nested in Loop-3 positions")

        prefix_update_ranks = tuple(
            torch.nonzero(logical < self.packed_tokens, as_tuple=False).flatten()
            for logical in (positions3, positions4)
        )
        prefix_positions = tuple(
            logical.index_select(0, compact)
            for logical, compact in zip(
                (positions3, positions4), prefix_update_ranks, strict=True
            )
        )
        if self.packed_tokens:
            self._install_sparse_streams(
                packed=self._packed_keys,
                absolute=self._absolute_keys,
                targets=(loop3_keys, loop4_keys),
                prefix_positions=prefix_positions,  # type: ignore[arg-type]
                prefix_update_ranks=prefix_update_ranks,  # type: ignore[arg-type]
                pack=lambda values, group: _pack_key(
                    values, group, self.reader_backend
                ),
            )
            self._install_sparse_streams(
                packed=self._packed_values,
                absolute=self._absolute_values,
                targets=(loop3_values, loop4_values),
                prefix_positions=prefix_positions,  # type: ignore[arg-type]
                prefix_update_ranks=prefix_update_ranks,  # type: ignore[arg-type]
                pack=lambda values, group: _pack_value(
                    values, group, self.reader_backend
                ),
            )

        result = CrossLoopKVLayer()
        result.group_size = self.group_size
        result.residual_length = self.residual_length
        result.quantized_tokens = self.packed_tokens
        result._shape_prefix = (self._shape[0], self._shape[1], self._shape[3])
        result._dtype = self._dtype
        result._device = self._device
        result.absolute_late_overrides = self.absolute_late_overrides
        result.absolute_override_dtype = self.absolute_override_dtype
        result.reader_backend = self.reader_backend
        result._packed_keys = self._packed_keys
        result._packed_values = self._packed_values
        result._absolute_keys = self._absolute_keys
        result._absolute_values = self._absolute_values
        result.override_indices = prefix_positions  # type: ignore[assignment]
        result.rank_maps = tuple(
            _rank_map(self.packed_tokens, logical, self._device)
            for logical in prefix_positions
        )  # type: ignore[assignment]

        tail_keys = list(self._tail_keys)
        tail_values = list(self._tail_values)
        for logical, target_key, target_value in (
            (positions3, loop3_keys, loop3_values),
            (positions4, loop4_keys, loop4_values),
        ):
            tail_key = tail_keys[-1].clone()
            tail_value = tail_values[-1].clone()
            compact = torch.nonzero(
                logical >= self.packed_tokens, as_tuple=False
            ).flatten()
            if compact.numel():
                tail_positions = (
                    logical.index_select(0, compact).long() - self.packed_tokens
                )
                tail_key.index_copy_(
                    2, tail_positions, target_key.index_select(2, compact)
                )
                tail_value.index_copy_(
                    2, tail_positions, target_value.index_select(2, compact)
                )
            tail_keys.append(tail_key)
            tail_values.append(tail_value)
        tail_masks = []
        for logical in (positions3, positions4):
            mask = torch.zeros(
                (1, self._shape[2] - self.packed_tokens),
                dtype=torch.bool,
                device=self._device,
            )
            selected = logical[logical >= self.packed_tokens].long() - self.packed_tokens
            if selected.numel():
                mask[:, selected] = True
            tail_masks.append(mask)
        result._install_tail_storage(
            tail_keys, tail_values, tail_masks[0], tail_masks[1]
        )
        return result


class CrossLoopKVLayer:
    """One physical layer's packed recurrent cache."""

    def __init__(self) -> None:
        self.group_size = 0
        self.residual_length = 0
        self.quantized_tokens = 0
        self.absolute_late_overrides = False
        self.absolute_override_dtype = "int4"
        self.reader_backend = "triton_row"
        self._packed_keys: list[PackedKV | None] = [None] * 4
        self._packed_values: list[PackedKV | None] = [None] * 4
        self._absolute_keys: list[torch.Tensor | None] = [None] * 4
        self._absolute_values: list[torch.Tensor | None] = [None] * 4
        self._tail_keys: list[torch.Tensor] = []
        self._tail_values: list[torch.Tensor] = []
        self._tail_active: tuple[torch.Tensor, torch.Tensor] = (
            torch.empty(0, dtype=torch.bool),
            torch.empty(0, dtype=torch.bool),
        )
        self._tail_length = 0
        self._tail_capacity = 0
        self.override_indices: tuple[torch.Tensor, torch.Tensor]
        self.rank_maps: tuple[torch.Tensor, torch.Tensor]
        self._shape_prefix: tuple[int, int, int] = (0, 0, 0)
        self._dtype: torch.dtype = torch.float16
        self._device = torch.device("cpu")
        self._split_kv_workspaces: dict[
            tuple[int, int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}

    def _split_kv_workspace(
        self, total_rows: int, *, split_rows: int = 128
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reuse per-layer online-softmax partials across decode steps."""

        from .kernels.quantization import _split_kv_geometry

        batch, heads, channels = self._shape_prefix
        num_splits, _ = _split_kv_geometry(total_rows, split_rows=split_rows)
        key = (batch * heads, num_splits, channels)
        workspace = self._split_kv_workspaces.get(key)
        if workspace is None:
            maximum = torch.empty(
                (batch * heads, num_splits),
                device=self._device,
                dtype=torch.float32,
            )
            denominator = torch.empty_like(maximum)
            accumulator = torch.empty(
                (batch * heads, num_splits, channels),
                device=self._device,
                dtype=torch.float32,
            )
            workspace = (maximum, denominator, accumulator)
            self._split_kv_workspaces[key] = workspace
        return workspace

    def _install_tail_storage(
        self,
        keys: Sequence[torch.Tensor],
        values: Sequence[torch.Tensor],
        active3: torch.Tensor,
        active4: torch.Tensor,
    ) -> None:
        length = int(keys[0].shape[2])
        if any(int(tensor.shape[2]) != length for tensor in (*keys, *values)):
            raise ValueError("tail K/V streams must share their token length")
        self._tail_keys = [
            tensor.clone(memory_format=torch.contiguous_format) for tensor in keys
        ]
        self._tail_values = [
            tensor.clone(memory_format=torch.contiguous_format) for tensor in values
        ]
        self._tail_active = (active3.clone(), active4.clone())
        self._tail_length = length
        self._tail_capacity = length

    def _ensure_tail_capacity(self, required: int) -> None:
        required = int(required)
        if required <= self._tail_capacity:
            return
        growth = min(self.group_size, 16)
        limit = self.residual_length + self.group_size
        capacity = min(limit, max(required, self._tail_capacity + growth))
        self._resize_tail_capacity(capacity)

    def _resize_tail_capacity(self, capacity: int) -> None:
        capacity = int(capacity)
        if capacity < self._tail_length:
            raise ValueError("tail capacity cannot be smaller than its logical length")
        if capacity == self._tail_capacity:
            return
        batch, heads, channels = self._shape_prefix
        grown_keys = [
            torch.empty(
                (batch, heads, capacity, channels),
                device=self._device,
                dtype=self._dtype,
            )
            for _ in range(4)
        ]
        grown_values = [torch.empty_like(tensor) for tensor in grown_keys]
        if self._tail_length:
            for target, source in zip(grown_keys, self._tail_keys, strict=True):
                target[:, :, : self._tail_length, :].copy_(
                    source[:, :, : self._tail_length, :]
                )
            for target, source in zip(grown_values, self._tail_values, strict=True):
                target[:, :, : self._tail_length, :].copy_(
                    source[:, :, : self._tail_length, :]
                )
        grown_active = [
            torch.zeros((batch, capacity), device=self._device, dtype=torch.bool)
            for _ in range(2)
        ]
        if self._tail_length:
            for target, source in zip(grown_active, self._tail_active, strict=True):
                target[:, : self._tail_length].copy_(source[:, : self._tail_length])
        self._tail_keys = grown_keys
        self._tail_values = grown_values
        self._tail_active = (grown_active[0], grown_active[1])
        self._tail_capacity = capacity

    @classmethod
    def from_prefill(
        cls,
        keys: Sequence[torch.Tensor],
        values: Sequence[torch.Tensor],
        *,
        mask_loop3: torch.Tensor,
        mask_loop4: torch.Tensor,
        group_size: int = 64,
        residual_length: int = 64,
        absolute_late_overrides: bool = False,
        absolute_override_dtype: str = "int4",
        reader_backend: str = "triton_row",
    ) -> "CrossLoopKVLayer":
        batch, heads, tokens, channels = _validate_loop_tensors(keys, values)
        group_size = int(group_size)
        residual_length = int(residual_length)
        if group_size <= 0 or residual_length < 0:
            raise ValueError("invalid group_size or residual_length")
        if mask_loop3.shape != (batch, tokens) or mask_loop4.shape != (batch, tokens):
            raise ValueError("prefill masks must have shape [1, tokens]")
        mask_loop3 = mask_loop3.to(device=keys[0].device, dtype=torch.bool)
        mask_loop4 = mask_loop4.to(device=keys[0].device, dtype=torch.bool)
        if not bool(torch.all(~mask_loop4 | mask_loop3).item()):
            raise ValueError("Loop-4 overrides must be nested in Loop-3 overrides")

        packed_tokens = max(0, ((tokens - residual_length) // group_size) * group_size)
        result = cls()
        result.group_size = group_size
        result.residual_length = residual_length
        result.quantized_tokens = packed_tokens
        result._shape_prefix = (batch, heads, channels)
        result._dtype = keys[0].dtype
        result._device = keys[0].device
        result.absolute_late_overrides = bool(absolute_late_overrides)
        if absolute_override_dtype not in {"int4", "bf16"}:
            raise ValueError("absolute_override_dtype must be 'int4' or 'bf16'")
        if absolute_override_dtype == "bf16" and not result.absolute_late_overrides:
            raise ValueError("bf16 absolute overrides require absolute_late_overrides=True")
        result.absolute_override_dtype = absolute_override_dtype
        if reader_backend not in {"kivi_outer", "triton_row"}:
            raise ValueError("unknown KV reader backend")
        if reader_backend == "kivi_outer" and result.absolute_late_overrides:
            raise ValueError("kivi_outer does not support absolute late overrides")
        result.reader_backend = reader_backend

        if packed_tokens:
            result._encode_initial_prefix(
                [tensor[:, :, :packed_tokens, :] for tensor in keys],
                [tensor[:, :, :packed_tokens, :] for tensor in values],
                mask_loop3[:, :packed_tokens],
                mask_loop4[:, :packed_tokens],
            )
        else:
            empty = torch.empty(0, dtype=torch.int32, device=result._device)
            result.override_indices = (empty.clone(), empty.clone())
            rank = torch.empty(0, dtype=torch.int32, device=result._device)
            result.rank_maps = (rank.clone(), rank.clone())

        result._install_tail_storage(
            [tensor[:, :, packed_tokens:, :] for tensor in keys],
            [tensor[:, :, packed_tokens:, :] for tensor in values],
            mask_loop3[:, packed_tokens:].clone(),
            mask_loop4[:, packed_tokens:].clone(),
        )
        return result

    @classmethod
    def from_prefill_updates(
        cls,
        *,
        loop1_key: torch.Tensor,
        loop1_value: torch.Tensor,
        loop2_key: torch.Tensor,
        loop2_value: torch.Tensor,
        loop3_positions: torch.Tensor,
        loop3_keys: torch.Tensor,
        loop3_values: torch.Tensor,
        loop4_positions: torch.Tensor,
        loop4_keys: torch.Tensor,
        loop4_values: torch.Tensor,
        group_size: int = 64,
        residual_length: int = 64,
        absolute_late_overrides: bool = False,
        absolute_override_dtype: str = "int4",
        reader_backend: str = "triton_row",
    ) -> "CrossLoopKVLayer":
        """Build physical storage without ever constructing full Loop-3/4 K/V."""

        shape = tuple(int(value) for value in loop1_key.shape)
        if len(shape) != 4 or shape[0] != 1:
            raise ValueError("dense prefill K/V must have shape [1, heads, tokens, channels]")
        dense = (loop1_key, loop1_value, loop2_key, loop2_value)
        if any(tensor.shape != loop1_key.shape for tensor in dense):
            raise ValueError("Loop-1/2 K/V tensors must share a shape")
        if any(tensor.dtype != loop1_key.dtype or tensor.device != loop1_key.device for tensor in dense):
            raise ValueError("Loop-1/2 K/V tensors must share dtype and device")
        batch, heads, tokens, channels = shape
        positions = []
        update_pairs = (
            (loop3_positions, loop3_keys, loop3_values),
            (loop4_positions, loop4_keys, loop4_values),
        )
        for logical, key, value in update_pairs:
            logical = logical.to(device=loop1_key.device, dtype=torch.int32).contiguous()
            if logical.ndim != 1 or (logical.numel() > 1 and not bool(torch.all(logical[1:] > logical[:-1]).item())):
                raise ValueError("late-loop positions must be strictly sorted")
            if logical.numel() and (int(logical.min().item()) < 0 or int(logical.max().item()) >= tokens):
                raise IndexError("late-loop update is outside the prompt")
            expected = (batch, heads, int(logical.numel()), channels)
            if tuple(key.shape) != expected or value.shape != key.shape:
                raise ValueError("late-loop compact K/V shape differs from its positions")
            positions.append(logical)
        positions3, positions4 = positions
        if positions4.numel():
            ranks4 = torch.searchsorted(positions3, positions4)
            valid_rank = ranks4 < positions3.numel()
            matched = torch.zeros_like(valid_rank)
            matched[valid_rank] = positions3[ranks4[valid_rank]] == positions4[valid_rank]
            if not bool(matched.all().item()):
                raise ValueError("Loop-4 positions must be nested in Loop-3 positions")

        group_size = int(group_size)
        residual_length = int(residual_length)
        if group_size <= 0 or residual_length < 0:
            raise ValueError("invalid group_size or residual_length")
        packed_tokens = max(0, ((tokens - residual_length) // group_size) * group_size)
        result = cls()
        result.group_size = group_size
        result.residual_length = residual_length
        result.quantized_tokens = packed_tokens
        result._shape_prefix = (batch, heads, channels)
        result._dtype = loop1_key.dtype
        result._device = loop1_key.device
        result.absolute_late_overrides = bool(absolute_late_overrides)
        if absolute_override_dtype not in {"int4", "bf16"}:
            raise ValueError("absolute_override_dtype must be 'int4' or 'bf16'")
        if absolute_override_dtype == "bf16" and not result.absolute_late_overrides:
            raise ValueError("bf16 absolute overrides require absolute_late_overrides=True")
        result.absolute_override_dtype = absolute_override_dtype
        if reader_backend not in {"kivi_outer", "triton_row"}:
            raise ValueError("unknown KV reader backend")
        if reader_backend == "kivi_outer" and result.absolute_late_overrides:
            raise ValueError("kivi_outer does not support absolute late overrides")
        result.reader_backend = reader_backend

        prefix_positions: list[torch.Tensor] = []
        prefix_update_ranks: list[torch.Tensor] = []
        for logical in (positions3, positions4):
            compact_ranks = torch.nonzero(logical < packed_tokens, as_tuple=False).flatten()
            prefix_update_ranks.append(compact_ranks)
            prefix_positions.append(logical.index_select(0, compact_ranks))
        result.override_indices = tuple(prefix_positions)  # type: ignore[assignment]
        result.rank_maps = tuple(
            _rank_map(packed_tokens, logical, result._device) for logical in prefix_positions
        )  # type: ignore[assignment]

        if packed_tokens:
            anchor_key = loop1_key[:, :, :packed_tokens, :]
            anchor_value = loop1_value[:, :, :packed_tokens, :]
            result._packed_keys[0] = _pack_key(anchor_key, group_size, reader_backend)
            result._packed_values[0] = _pack_value(anchor_value, group_size, reader_backend)
            reconstructed_key = _decode_packed(result._packed_keys[0])
            reconstructed_value = _decode_packed(result._packed_values[0])
            delta2_key = loop2_key[:, :, :packed_tokens, :] - reconstructed_key
            delta2_value = loop2_value[:, :, :packed_tokens, :] - reconstructed_value
            result._packed_keys[1] = _pack_key(delta2_key, group_size, reader_backend)
            result._packed_values[1] = _pack_value(delta2_value, group_size, reader_backend)
            reconstructed_key = reconstructed_key + _decode_packed(result._packed_keys[1])
            reconstructed_value = reconstructed_value + _decode_packed(result._packed_values[1])

            target_keys = (loop3_keys, loop4_keys)
            target_values = (loop3_values, loop4_values)
            decoded3_key: torch.Tensor | None = None
            decoded3_value: torch.Tensor | None = None
            for stream_index in (2, 3):
                logical = prefix_positions[stream_index - 2]
                compact = prefix_update_ranks[stream_index - 2]
                if not logical.numel():
                    continue
                target_key = target_keys[stream_index - 2].index_select(2, compact)
                target_value = target_values[stream_index - 2].index_select(2, compact)
                if result.absolute_late_overrides:
                    if result.absolute_override_dtype == "bf16":
                        result._absolute_keys[stream_index] = target_key.to(
                            torch.bfloat16
                        ).contiguous()
                        result._absolute_values[stream_index] = target_value.to(
                            torch.bfloat16
                        ).contiguous()
                    else:
                        result._packed_keys[stream_index] = _pack_key(
                            target_key, group_size, reader_backend
                        )
                        result._packed_values[stream_index] = _pack_value(
                            target_value, group_size, reader_backend
                        )
                    continue
                base_key = reconstructed_key.index_select(2, logical.long())
                base_value = reconstructed_value.index_select(2, logical.long())
                if stream_index == 3:
                    assert decoded3_key is not None and decoded3_value is not None
                    ranks_in_loop3 = torch.searchsorted(prefix_positions[0], logical).long()
                    base_key = base_key + decoded3_key.index_select(2, ranks_in_loop3)
                    base_value = base_value + decoded3_value.index_select(2, ranks_in_loop3)
                result._packed_keys[stream_index] = _pack_key(
                    target_key - base_key,
                    group_size,
                    reader_backend,
                )
                result._packed_values[stream_index] = _pack_value(
                    target_value - base_value,
                    group_size,
                    reader_backend,
                )
                if stream_index == 2:
                    decoded3_key = _decode_packed(result._packed_keys[2])
                    decoded3_value = _decode_packed(result._packed_values[2])

        tail_keys = [
            loop1_key[:, :, packed_tokens:, :].clone(),
            loop2_key[:, :, packed_tokens:, :].clone(),
        ]
        tail_values = [
            loop1_value[:, :, packed_tokens:, :].clone(),
            loop2_value[:, :, packed_tokens:, :].clone(),
        ]
        for logical, target_key, target_value in update_pairs:
            tail_key = tail_keys[-1].clone()
            tail_value = tail_values[-1].clone()
            compact = torch.nonzero(logical >= packed_tokens, as_tuple=False).flatten()
            if compact.numel():
                tail_positions = logical.index_select(0, compact).long() - packed_tokens
                tail_key.index_copy_(2, tail_positions, target_key.index_select(2, compact))
                tail_value.index_copy_(2, tail_positions, target_value.index_select(2, compact))
            tail_keys.append(tail_key)
            tail_values.append(tail_value)
        tail_masks = []
        for logical in (positions3, positions4):
            mask = torch.zeros(
                (1, tokens - packed_tokens),
                dtype=torch.bool,
                device=result._device,
            )
            selected = logical[logical >= packed_tokens].long() - packed_tokens
            if selected.numel():
                mask[:, selected] = True
            tail_masks.append(mask)
        result._install_tail_storage(
            tail_keys,
            tail_values,
            tail_masks[0],
            tail_masks[1],
        )
        return result

    def _encode_initial_prefix(
        self,
        keys: Sequence[torch.Tensor],
        values: Sequence[torch.Tensor],
        mask3: torch.Tensor,
        mask4: torch.Tensor,
    ) -> None:
        tokens = int(keys[0].shape[2])
        self._packed_keys[0] = _pack_key(keys[0], self.group_size, self.reader_backend)
        self._packed_values[0] = _pack_value(values[0], self.group_size, self.reader_backend)
        reconstructed_k = _decode_packed(self._packed_keys[0])
        reconstructed_v = _decode_packed(self._packed_values[0])

        delta2_k = keys[1] - reconstructed_k
        delta2_v = values[1] - reconstructed_v
        self._packed_keys[1] = _pack_key(delta2_k, self.group_size, self.reader_backend)
        self._packed_values[1] = _pack_value(delta2_v, self.group_size, self.reader_backend)
        reconstructed_k = reconstructed_k + _decode_packed(self._packed_keys[1])
        reconstructed_v = reconstructed_v + _decode_packed(self._packed_values[1])

        positions3 = torch.nonzero(mask3[0], as_tuple=False).flatten().to(torch.int32)
        positions4 = torch.nonzero(mask4[0], as_tuple=False).flatten().to(torch.int32)
        self.override_indices = (positions3, positions4)
        self.rank_maps = (
            _rank_map(tokens, positions3, self._device),
            _rank_map(tokens, positions4, self._device),
        )

        for stream_index, (target_k, target_v, positions) in enumerate(
            ((keys[2], values[2], positions3), (keys[3], values[3], positions4)),
            start=2,
        ):
            if positions.numel() == 0:
                continue
            selected = positions.long()
            if self.absolute_late_overrides:
                selected_key = target_k.index_select(2, selected)
                selected_value = target_v.index_select(2, selected)
                if self.absolute_override_dtype == "bf16":
                    self._absolute_keys[stream_index] = selected_key.to(
                        torch.bfloat16
                    ).contiguous()
                    self._absolute_values[stream_index] = selected_value.to(
                        torch.bfloat16
                    ).contiguous()
                else:
                    self._packed_keys[stream_index] = _pack_key(
                        selected_key, self.group_size, self.reader_backend
                    )
                    self._packed_values[stream_index] = _pack_value(
                        selected_value, self.group_size, self.reader_backend
                    )
                continue
            delta_k = target_k.index_select(2, selected) - reconstructed_k.index_select(2, selected)
            delta_v = target_v.index_select(2, selected) - reconstructed_v.index_select(2, selected)
            self._packed_keys[stream_index] = _pack_key(
                delta_k, self.group_size, self.reader_backend
            )
            self._packed_values[stream_index] = _pack_value(
                delta_v, self.group_size, self.reader_backend
            )
            decoded_k = _decode_packed(self._packed_keys[stream_index])
            decoded_v = _decode_packed(self._packed_values[stream_index])
            reconstructed_k = reconstructed_k.clone()
            reconstructed_v = reconstructed_v.clone()
            reconstructed_k.index_add_(2, selected, decoded_k)
            reconstructed_v.index_add_(2, selected, decoded_v)

    @property
    def tail_tokens(self) -> int:
        return self._tail_length

    @property
    def token_count(self) -> int:
        return self.quantized_tokens + self.tail_tokens

    @property
    def storage_bytes(self) -> int:
        packed = sum(
            stream.storage_bytes
            for stream in (*self._packed_keys, *self._packed_values)
            if stream is not None
        )
        absolute = sum(
            _tensor_bytes(stream)
            for stream in (*self._absolute_keys, *self._absolute_values)
            if stream is not None
        )
        tails = sum(_tensor_bytes(tensor) for tensor in (*self._tail_keys, *self._tail_values))
        indices = sum(
            _tensor_bytes(tensor)
            for tensor in (*self.override_indices, *self.rank_maps, *self._tail_active)
        )
        return packed + absolute + tails + indices

    @property
    def dense_bf16_bytes(self) -> int:
        batch, heads, channels = self._shape_prefix
        element_size = torch.empty((), dtype=self._dtype).element_size()
        return 4 * 2 * batch * heads * self.token_count * channels * element_size

    @property
    def storage_ratio(self) -> float:
        dense = self.dense_bf16_bytes
        return self.storage_bytes / dense if dense else 0.0

    def gather(self, loop: int, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        loop = int(loop)
        if loop not in range(4):
            raise ValueError("loop must be in [0, 3]")
        if indices.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        indices = indices.to(device=self._device, dtype=torch.long)
        if indices.numel() and (
            int(indices.min().item()) < 0 or int(indices.max().item()) >= self.token_count
        ):
            raise IndexError("cache index is outside the logical sequence")
        batch, heads, channels = self._shape_prefix
        output_shape = (batch, heads, int(indices.numel()), channels)
        output_k = torch.empty(output_shape, dtype=self._dtype, device=self._device)
        output_v = torch.empty_like(output_k)

        prefix_output_positions = torch.nonzero(
            indices < self.quantized_tokens,
            as_tuple=False,
        ).flatten()
        if prefix_output_positions.numel():
            prefix_indices = indices.index_select(0, prefix_output_positions)
            anchor_k = self._packed_keys[0]
            anchor_v = self._packed_values[0]
            assert anchor_k is not None and anchor_v is not None
            prefix_k = _decode_rows(anchor_k, prefix_indices)
            prefix_v = _decode_rows(anchor_v, prefix_indices)
            for stream_index in range(1, loop + 1):
                packed_k = self._packed_keys[stream_index]
                packed_v = self._packed_values[stream_index]
                absolute_k = self._absolute_keys[stream_index]
                absolute_v = self._absolute_values[stream_index]
                if (
                    packed_k is None
                    and packed_v is None
                    and absolute_k is None
                    and absolute_v is None
                ):
                    continue
                if stream_index == 1:
                    assert packed_k is not None and packed_v is not None
                    prefix_k = prefix_k + _decode_rows(packed_k, prefix_indices)
                    prefix_v = prefix_v + _decode_rows(packed_v, prefix_indices)
                    continue
                rank_map = self.rank_maps[stream_index - 2]
                ranks = rank_map.index_select(0, prefix_indices)
                active_output = torch.nonzero(ranks >= 0, as_tuple=False).flatten()
                if active_output.numel():
                    active_ranks = ranks.index_select(0, active_output).long()
                    if absolute_k is not None or absolute_v is not None:
                        assert absolute_k is not None and absolute_v is not None
                        update_k = absolute_k.index_select(2, active_ranks).to(
                            self._dtype
                        )
                        update_v = absolute_v.index_select(2, active_ranks).to(
                            self._dtype
                        )
                    else:
                        assert packed_k is not None and packed_v is not None
                        update_k = _decode_rows(packed_k, active_ranks)
                        update_v = _decode_rows(packed_v, active_ranks)
                    if self.absolute_late_overrides:
                        prefix_k.index_copy_(2, active_output, update_k)
                        prefix_v.index_copy_(2, active_output, update_v)
                    else:
                        prefix_k.index_add_(2, active_output, update_k)
                        prefix_v.index_add_(2, active_output, update_v)
            output_k.index_copy_(2, prefix_output_positions, prefix_k)
            output_v.index_copy_(2, prefix_output_positions, prefix_v)

        tail_output_positions = torch.nonzero(
            indices >= self.quantized_tokens,
            as_tuple=False,
        ).flatten()
        if tail_output_positions.numel():
            tail_indices = indices.index_select(0, tail_output_positions) - self.quantized_tokens
            output_k.index_copy_(
                2,
                tail_output_positions,
                self._tail_keys[loop].index_select(2, tail_indices),
            )
            output_v.index_copy_(
                2,
                tail_output_positions,
                self._tail_values[loop].index_select(2, tail_indices),
            )
        return output_k, output_v

    def qk(
        self,
        loop: int,
        query: torch.Tensor,
        indices: torch.Tensor,
        *,
        validate_indices: bool = True,
    ) -> torch.Tensor:
        """Compute QK directly from packed anchor/deltas plus the BF16 tail."""

        from .kernels.quantization import triton_packed_qk

        loop = int(loop)
        if loop not in range(4):
            raise ValueError("loop must be in [0, 3]")
        batch, heads, channels = self._shape_prefix
        if tuple(query.shape) != (batch, heads, 1, channels):
            raise ValueError("query shape differs from this cache layer")
        logical = _normalize_logical_indices(
            indices,
            batch=batch,
            heads=heads,
            tokens=self.token_count,
            device=self._device,
            validate_bounds=validate_indices,
        )
        if (
            self.reader_backend == "triton_row"
            and self.quantized_tokens
            and query.is_cuda
        ):
            from .kernels.quantization import triton_cross_loop_qk

            return triton_cross_loop_qk(
                query,
                tuple(self._packed_keys),
                self.rank_maps,
                self._tail_keys[loop],
                logical,
                loop=loop,
                absolute_keys=tuple(self._absolute_keys),
                absolute_late_overrides=self.absolute_late_overrides,
            )
        selected = int(logical.shape[-1])
        if self.reader_backend == "kivi_outer" and self.quantized_tokens:
            packed_keys = tuple(self._packed_keys)
            if not all(
                stream is None or isinstance(stream, KiviPacked4Bit)
                for stream in packed_keys
            ):
                raise RuntimeError("kivi_outer cache contains a non-KIVI key stream")

            if (
                query.is_cuda
                and selected * 2 <= self.quantized_tokens
                and all(stream is not None for stream in packed_keys)
            ):
                output = kivi_fused_cross_loop_qk(
                    query,
                    packed_keys,  # type: ignore[arg-type]
                    self.rank_maps,
                    logical,
                    loop=loop,
                    extension=load_kivi_outer_extension(),
                )
            else:
                def gemv(inputs: torch.Tensor, packed: KiviPacked4Bit) -> torch.Tensor:
                    if inputs.is_cuda:
                        return kivi_outer_gemv(
                            inputs,
                            packed,
                            extension=load_kivi_outer_extension(),
                        )
                    return outer_matmul_reference(inputs, packed)

                selected_gemv = None
                if query.is_cuda and selected * 2 <= self.quantized_tokens:
                    extension = load_kivi_outer_extension()

                    def selected_gemv(
                        inputs: torch.Tensor,
                        packed: KiviPacked4Bit,
                        selected_indices: torch.Tensor,
                    ) -> torch.Tensor:
                        return kivi_selected_qk(
                            inputs,
                            packed,
                            selected_indices,
                            extension=extension,
                        )

                output = kivi_cross_loop_qk(
                    query,
                    packed_keys,  # type: ignore[arg-type]
                    self.rank_maps,
                    logical,
                    loop=loop,
                    gemv=gemv,
                    selected_gemv=selected_gemv,
                )
        else:
            output = torch.zeros(
                (batch, heads, 1, selected),
                device=self._device,
                dtype=torch.float32,
            )
        prefix = logical < self.quantized_tokens
        if self.quantized_tokens and self.reader_backend == "triton_row":
            for stream_index in range(loop + 1):
                packed = self._packed_keys[stream_index]
                absolute = self._absolute_keys[stream_index]
                if packed is None and absolute is None:
                    continue
                if stream_index < 2:
                    active = prefix
                    packed_indices = logical
                else:
                    safe_logical = torch.where(prefix, logical, torch.zeros_like(logical))
                    ranks = self.rank_maps[stream_index - 2][safe_logical.long()]
                    active = prefix & (ranks >= 0)
                    packed_indices = ranks
                safe = torch.where(active, packed_indices, torch.zeros_like(packed_indices))
                if absolute is not None:
                    gathered = torch.gather(
                        absolute,
                        2,
                        safe.long().unsqueeze(-1).expand(-1, -1, -1, channels),
                    )
                    contribution = torch.matmul(
                        query.float(), gathered.float().transpose(-1, -2)
                    )
                else:
                    assert packed is not None
                    contribution = triton_packed_qk(query, packed, safe)
                contribution *= active[:, :, None, :]
                if self.absolute_late_overrides and stream_index >= 2:
                    output = torch.where(active[:, :, None, :], contribution, output)
                else:
                    output += contribution

        tail = logical >= self.quantized_tokens
        if self.tail_tokens:
            safe_tail = torch.where(
                tail,
                logical - self.quantized_tokens,
                torch.zeros_like(logical),
            )
            tail_key = torch.gather(
                self._tail_keys[loop],
                2,
                safe_tail.long().unsqueeze(-1).expand(-1, -1, -1, channels),
            )
            tail_logits = torch.matmul(query.float(), tail_key.float().transpose(-1, -2))
            output += tail_logits * tail[:, :, None, :]
        return output

    def pv(
        self,
        loop: int,
        weights: torch.Tensor,
        indices: torch.Tensor,
        *,
        validate_indices: bool = True,
    ) -> torch.Tensor:
        """Compute probability-weighted V directly from physical cross-loop storage."""

        from .kernels.quantization import triton_packed_pv

        loop = int(loop)
        if loop not in range(4):
            raise ValueError("loop must be in [0, 3]")
        batch, heads, channels = self._shape_prefix
        logical = _normalize_logical_indices(
            indices,
            batch=batch,
            heads=heads,
            tokens=self.token_count,
            device=self._device,
            validate_bounds=validate_indices,
        )
        selected = int(logical.shape[-1])
        if tuple(weights.shape) != (batch, heads, 1, selected):
            raise ValueError("weights must match [batch, heads, 1, selected]")
        if (
            self.reader_backend == "triton_row"
            and self.quantized_tokens
            and weights.is_cuda
        ):
            from .kernels.quantization import triton_cross_loop_pv

            return triton_cross_loop_pv(
                weights,
                tuple(self._packed_values),
                self.rank_maps,
                self._tail_values[loop],
                logical,
                loop=loop,
                absolute_values=tuple(self._absolute_values),
                absolute_late_overrides=self.absolute_late_overrides,
            )
        if self.reader_backend == "kivi_outer" and self.quantized_tokens:
            packed_values = tuple(self._packed_values)
            if not all(
                stream is None or isinstance(stream, KiviPacked4Bit)
                for stream in packed_values
            ):
                raise RuntimeError("kivi_outer cache contains a non-KIVI value stream")

            if (
                weights.is_cuda
                and selected * 2 <= self.quantized_tokens
                and all(stream is not None for stream in packed_values)
            ):
                output = kivi_fused_cross_loop_pv(
                    weights,
                    packed_values,  # type: ignore[arg-type]
                    self.rank_maps,
                    logical,
                    loop=loop,
                    extension=load_kivi_outer_extension(),
                )
            else:
                def gemv(inputs: torch.Tensor, packed: KiviPacked4Bit) -> torch.Tensor:
                    if inputs.is_cuda:
                        return kivi_outer_gemv(
                            inputs,
                            packed,
                            extension=load_kivi_outer_extension(),
                        )
                    return outer_matmul_reference(inputs, packed)

                selected_gemv = None
                if weights.is_cuda and selected * 2 <= self.quantized_tokens:
                    extension = load_kivi_outer_extension()

                    def selected_gemv(
                        inputs: torch.Tensor,
                        packed: KiviPacked4Bit,
                        selected_indices: torch.Tensor,
                    ) -> torch.Tensor:
                        return kivi_selected_pv(
                            inputs,
                            packed,
                            selected_indices,
                            extension=extension,
                        )

                output = kivi_cross_loop_pv(
                    weights,
                    packed_values,  # type: ignore[arg-type]
                    self.rank_maps,
                    logical,
                    loop=loop,
                    gemv=gemv,
                    selected_gemv=selected_gemv,
                )
        else:
            output = torch.zeros(
                (batch, heads, 1, channels),
                device=self._device,
                dtype=torch.float32,
            )
        prefix = logical < self.quantized_tokens
        if self.quantized_tokens and self.reader_backend == "triton_row":
            for stream_index in range(loop + 1):
                packed = self._packed_values[stream_index]
                absolute = self._absolute_values[stream_index]
                if packed is None and absolute is None:
                    continue
                if stream_index < 2:
                    active = prefix
                    packed_indices = logical
                else:
                    safe_logical = torch.where(prefix, logical, torch.zeros_like(logical))
                    ranks = self.rank_maps[stream_index - 2][safe_logical.long()]
                    active = prefix & (ranks >= 0)
                    packed_indices = ranks
                safe = torch.where(active, packed_indices, torch.zeros_like(packed_indices))
                masked_weights = weights.float() * active[:, :, None, :]
                if absolute is not None:
                    gathered = torch.gather(
                        absolute,
                        2,
                        safe.long().unsqueeze(-1).expand(-1, -1, -1, channels),
                    )
                    contribution = torch.matmul(masked_weights, gathered.float())
                else:
                    assert packed is not None
                    contribution = triton_packed_pv(masked_weights, packed, safe)
                if self.absolute_late_overrides and stream_index >= 2:
                    previous = self.pv(
                        stream_index - 1,
                        masked_weights,
                        logical,
                        validate_indices=False,
                    )
                    output += contribution - previous
                else:
                    output += contribution

        tail = logical >= self.quantized_tokens
        if self.tail_tokens:
            safe_tail = torch.where(
                tail,
                logical - self.quantized_tokens,
                torch.zeros_like(logical),
            )
            tail_value = torch.gather(
                self._tail_values[loop],
                2,
                safe_tail.long().unsqueeze(-1).expand(-1, -1, -1, channels),
            )
            output += torch.matmul(weights.float() * tail[:, :, None, :], tail_value.float())
        return output

    def cached_mass_sparse_attention_delta(
        self,
        loop: int,
        *,
        query: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        current_position: int,
        indices: torch.Tensor,
        valid: torch.Tensor,
        source_selected_output: torch.Tensor,
        global_mass: torch.Tensor,
        scaling: float,
    ) -> torch.Tensor:
        """Return the cached-mass late-loop attention correction."""

        loop = int(loop)
        if loop not in (2, 3):
            raise ValueError("cached-mass sparse attention is only valid for Loop 3/4")
        batch, heads, channels = self._shape_prefix
        expected_vector = (batch, heads, 1, channels)
        if any(
            tuple(tensor.shape) != expected_vector
            for tensor in (query, current_key, current_value, source_selected_output)
        ):
            raise ValueError("attention vectors differ from this cache layer")
        logical = _normalize_logical_indices(
            indices,
            batch=batch,
            heads=heads,
            tokens=max(self.token_count, int(current_position) + 1),
            device=self._device,
            validate_bounds=False,
        )
        if tuple(valid.shape) != tuple(logical.shape):
            raise ValueError("valid mask must match selected indices")
        valid = valid.to(device=self._device, dtype=torch.bool).contiguous()
        if tuple(global_mass.shape) != (batch, heads, 1, 1):
            raise ValueError("global mass must have shape [batch, heads, 1, 1]")

        if (
            self.reader_backend == "triton_row"
            and self.quantized_tokens
            and query.is_cuda
        ):
            from .kernels.quantization import triton_cross_loop_sparse_attention_delta

            schedule = _packed_attention_schedule("sparse", int(logical.shape[-1]))

            return triton_cross_loop_sparse_attention_delta(
                query=query,
                current_key=current_key,
                current_value=current_value,
                packed_keys=tuple(self._packed_keys),
                packed_values=tuple(self._packed_values),
                absolute_keys=tuple(self._absolute_keys),
                absolute_values=tuple(self._absolute_values),
                rank_maps=self.rank_maps,
                tail_key=self._tail_keys[loop],
                tail_value=self._tail_values[loop],
                indices=logical,
                valid=valid,
                source_selected_output=source_selected_output,
                global_mass=global_mass,
                current_position=int(current_position),
                scaling=float(scaling),
                loop=loop,
                absolute_late_overrides=self.absolute_late_overrides,
                split_kv=schedule == "split_kv",
                workspace=(
                    self._split_kv_workspace(int(logical.shape[-1]))
                    if schedule == "split_kv"
                    else None
                ),
            )

        if (
            self.reader_backend == "kivi_outer"
            and self.quantized_tokens
            and query.is_cuda
            and all(stream is not None for stream in self._packed_keys)
            and all(stream is not None for stream in self._packed_values)
        ):
            return kivi_fused_sparse_attention_delta(
                query=query,
                current_key=current_key,
                current_value=current_value,
                key_streams=tuple(self._packed_keys),  # type: ignore[arg-type]
                value_streams=tuple(self._packed_values),  # type: ignore[arg-type]
                rank_maps=self.rank_maps,
                tail_key=self._tail_keys[loop],
                tail_value=self._tail_values[loop],
                indices=logical,
                valid=valid,
                source_selected_output=source_selected_output,
                global_mass=global_mass,
                current_position=int(current_position),
                scaling=float(scaling),
                loop=loop,
                extension=load_kivi_outer_extension(),
            )

        is_current = valid & (logical == int(current_position))
        safe = torch.where(is_current, torch.zeros_like(logical), logical)
        logits = self.qk(loop, query, safe, validate_indices=False)
        current_logits = torch.matmul(
            query.float(), current_key.float().transpose(-1, -2)
        )
        logits = torch.where(is_current[:, :, None, :], current_logits, logits)
        logits = (logits * float(scaling)).masked_fill(
            ~valid[:, :, None, :],
            float("-inf"),
        )
        weights = torch.softmax(logits, dim=-1, dtype=torch.float32)
        target = self.pv(
            loop,
            weights * (~is_current)[:, :, None, :],
            safe,
            validate_indices=False,
        )
        target += torch.sum(
            weights * is_current[:, :, None, :],
            dim=-1,
            keepdim=True,
        ) * current_value.float()
        return global_mass.float() * (target - source_selected_output.float())

    def dense_source_attention(
        self,
        loop: int,
        *,
        query: torch.Tensor,
        current_key: torch.Tensor,
        current_value: torch.Tensor,
        scaling: float,
        return_logits: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run Loop-1/2 attention without materializing the packed cache."""

        loop = int(loop)
        if loop not in (0, 1):
            raise ValueError("dense source attention is only valid for Loop 1/2")
        batch, heads, channels = self._shape_prefix
        expected = (batch, heads, 1, channels)
        if any(
            tuple(tensor.shape) != expected
            for tensor in (query, current_key, current_value)
        ):
            raise ValueError("attention vectors differ from this cache layer")
        schedule = _packed_attention_schedule("dense", self.token_count)
        if (
            self.reader_backend == "kivi_outer"
            and self.quantized_tokens
            and query.is_cuda
            and all(stream is not None for stream in self._packed_keys)
            and all(stream is not None for stream in self._packed_values)
        ):
            return kivi_fused_dense_source_attention(
                query=query,
                current_key=current_key,
                current_value=current_value,
                key_streams=tuple(self._packed_keys),  # type: ignore[arg-type]
                value_streams=tuple(self._packed_values),  # type: ignore[arg-type]
                rank_maps=self.rank_maps,
                tail_key=self._tail_keys[loop][:, :, : self.tail_tokens, :],
                tail_value=self._tail_values[loop][:, :, : self.tail_tokens, :],
                scaling=float(scaling),
                loop=loop,
                extension=load_kivi_outer_extension(),
                return_logits=bool(return_logits),
            )

        if (
            self.reader_backend == "triton_row"
            and self.quantized_tokens
            and query.is_cuda
            and schedule in {"fused", "split_kv"}
        ):
            from .kernels.quantization import (
                triton_cross_loop_dense_source_attention,
            )

            return triton_cross_loop_dense_source_attention(
                query=query,
                current_key=current_key,
                current_value=current_value,
                packed_keys=tuple(self._packed_keys),
                packed_values=tuple(self._packed_values),
                tail_key=self._tail_keys[loop],
                tail_value=self._tail_values[loop],
                tail_tokens=self.tail_tokens,
                scaling=float(scaling),
                loop=loop,
                return_logits=bool(return_logits),
                split_kv=schedule == "split_kv",
                workspace=(
                    self._split_kv_workspace(self.token_count + 1)
                    if schedule == "split_kv"
                    else None
                ),
            )

        if self.quantized_tokens and query.is_cuda:
            logical = torch.arange(
                self.token_count,
                device=self._device,
                dtype=torch.int32,
            )[None, None].expand(batch, heads, -1).contiguous()
            past_logits = self.qk(
                loop,
                query,
                logical,
                validate_indices=False,
            )
            current_logits = torch.matmul(
                query.float(), current_key.float().transpose(-1, -2)
            )
            logits = torch.cat((past_logits, current_logits), dim=-1) * float(scaling)
            weights = torch.softmax(logits, dim=-1, dtype=torch.float32)
            output = self.pv(
                loop,
                weights[..., :-1],
                logical,
                validate_indices=False,
            )
            output += weights[..., -1:] * current_value.float()
            return output, logits if return_logits else None

        logical = torch.arange(self.token_count, device=self._device)
        past_key, past_value = self.gather(loop, logical)
        key = torch.cat((past_key, current_key), dim=2)
        value = torch.cat((past_value, current_value), dim=2)
        logits = (
            torch.matmul(query.float(), key.float().transpose(-1, -2))
            * float(scaling)
        )
        output = torch.matmul(
            torch.softmax(logits, dim=-1, dtype=torch.float32), value.float()
        )
        return output, logits if return_logits else None

    def append_decode_token(
        self,
        keys: Sequence[torch.Tensor],
        values: Sequence[torch.Tensor],
    ) -> None:
        shape = _validate_loop_tensors(keys, values)
        if shape[:2] != self._shape_prefix[:2] or shape[2] != 1 or shape[3] != self._shape_prefix[2]:
            raise ValueError("decode K/V shape does not match the cache")
        self._ensure_tail_capacity(self._tail_length + 1)
        target = self._tail_length
        for tail, incoming in zip(self._tail_keys, keys, strict=True):
            tail[:, :, target : target + 1, :].copy_(incoming)
        for tail, incoming in zip(self._tail_values, values, strict=True):
            tail[:, :, target : target + 1, :].copy_(incoming)
        for active in self._tail_active:
            active[:, target] = True
        self._tail_length += 1
        while self.tail_tokens >= self.residual_length + self.group_size:
            self._roll_one_group()

    def _roll_one_group(self) -> None:
        width = self.group_size
        block_keys = [tensor[:, :, :width, :] for tensor in self._tail_keys]
        block_values = [tensor[:, :, :width, :] for tensor in self._tail_values]
        block = CrossLoopKVLayer.from_prefill(
            block_keys,
            block_values,
            mask_loop3=self._tail_active[0][:, :width],
            mask_loop4=self._tail_active[1][:, :width],
            group_size=self.group_size,
            residual_length=0,
            absolute_late_overrides=self.absolute_late_overrides,
            absolute_override_dtype=self.absolute_override_dtype,
            reader_backend=self.reader_backend,
        )
        old_prefix = self.quantized_tokens
        for index in range(4):
            self._packed_keys[index] = _concat_packed(
                self._packed_keys[index],
                block._packed_keys[index],
            )
            self._packed_values[index] = _concat_packed(
                self._packed_values[index],
                block._packed_values[index],
            )
            self._absolute_keys[index] = _concat_optional_tensor(
                self._absolute_keys[index],
                block._absolute_keys[index],
            )
            self._absolute_values[index] = _concat_optional_tensor(
                self._absolute_values[index],
                block._absolute_values[index],
            )
        self.override_indices = tuple(
            torch.cat([current, addition + old_prefix], dim=0).to(torch.int32)
            for current, addition in zip(
                self.override_indices,
                block.override_indices,
                strict=True,
            )
        )
        self.quantized_tokens += width
        self.rank_maps = tuple(
            _rank_map(self.quantized_tokens, positions, self._device)
            for positions in self.override_indices
        )
        remaining = self._tail_length - width
        if remaining:
            for tensor in (*self._tail_keys, *self._tail_values):
                shifted = tensor[:, :, width : self._tail_length, :].clone()
                tensor[:, :, :remaining, :].copy_(shifted)
            for active in self._tail_active:
                shifted = active[:, width : self._tail_length].clone()
                active[:, :remaining].copy_(shifted)
        self._tail_length = remaining
        retained_capacity = min(
            self.residual_length + self.group_size,
            remaining + min(self.group_size, 16),
        )
        if self._tail_capacity > retained_capacity:
            self._resize_tail_capacity(retained_capacity)


class CrossLoopKVCache:
    """Layer-indexed owner for physical Cross-Loop KV storage."""

    def __init__(self, *, num_layers: int) -> None:
        num_layers = int(num_layers)
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.layers: list[CrossLoopKVLayer | object | None] = [None] * num_layers

    def set_layer(self, layer_index: int, layer: CrossLoopKVLayer | object) -> None:
        layer_index = int(layer_index)
        if layer_index not in range(len(self.layers)):
            raise IndexError("layer index is outside the cache")
        if self.layers[layer_index] is not None:
            raise RuntimeError("physical cache layer is already installed")
        self.layers[layer_index] = layer

    def physical_layout_audit(self) -> dict[str, int | list[str]]:
        """Report the sole persistent packed layout used by installed layers."""

        installed = [layer for layer in self.layers if layer is not None]
        backends = sorted({str(getattr(layer, "reader_backend")) for layer in installed})
        kivi_streams = 0
        row_streams = 0
        bf16_layers = 0
        for layer in installed:
            if getattr(layer, "storage_kind", None) == "bf16_cross_loop":
                bf16_layers += 1
                continue
            for stream in (*layer._packed_keys, *layer._packed_values):  # type: ignore[attr-defined]
                if isinstance(stream, KiviPacked4Bit):
                    kivi_streams += 1
                elif isinstance(stream, Packed4Bit):
                    row_streams += 1
        return {
            "installed_layers": len(installed),
            "reader_backends": backends,
            "kivi_packed_streams": kivi_streams,
            "row_packed_streams": row_streams,
            "bf16_cross_loop_layers": bf16_layers,
        }

    def gather(
        self,
        layer_index: int,
        loop: int,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_index = int(layer_index)
        if layer_index not in range(len(self.layers)):
            raise IndexError("layer index is outside the cache")
        layer = self.layers[layer_index]
        if layer is None:
            raise RuntimeError("physical cache layer has not been installed")
        return layer.gather(loop, indices)

    def qk(
        self,
        layer_index: int,
        loop: int,
        query: torch.Tensor,
        indices: torch.Tensor,
        *,
        validate_indices: bool = True,
    ) -> torch.Tensor:
        layer = self._require_layer(layer_index)
        return layer.qk(loop, query, indices, validate_indices=validate_indices)

    def pv(
        self,
        layer_index: int,
        loop: int,
        weights: torch.Tensor,
        indices: torch.Tensor,
        *,
        validate_indices: bool = True,
    ) -> torch.Tensor:
        layer = self._require_layer(layer_index)
        return layer.pv(loop, weights, indices, validate_indices=validate_indices)

    def cached_mass_sparse_attention_delta(
        self,
        layer_index: int,
        loop: int,
        **kwargs: object,
    ) -> torch.Tensor:
        layer = self._require_layer(layer_index)
        return layer.cached_mass_sparse_attention_delta(loop, **kwargs)  # type: ignore[arg-type]

    def dense_source_attention(
        self,
        layer_index: int,
        loop: int,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        layer = self._require_layer(layer_index)
        return layer.dense_source_attention(loop, **kwargs)  # type: ignore[arg-type]

    def _require_layer(self, layer_index: int) -> CrossLoopKVLayer:
        layer_index = int(layer_index)
        if layer_index not in range(len(self.layers)):
            raise IndexError("layer index is outside the cache")
        layer = self.layers[layer_index]
        if layer is None:
            raise RuntimeError("physical cache layer has not been installed")
        return layer  # type: ignore[return-value]

    @property
    def storage_bytes(self) -> int:
        return sum(layer.storage_bytes for layer in self.layers if layer is not None)
