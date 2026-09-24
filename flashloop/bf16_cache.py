"""Unquantized Cross-Loop KV storage used by component-efficiency ablations.

The production cache packs the same logical layout to 4 bit.  This reference
backend keeps Loop 1/2 in BF16 and stores only the selected absolute Loop 3/4
rows in BF16, allowing the quantization component to be isolated cleanly.
"""

from __future__ import annotations

from typing import Sequence

import torch


def _tensor_bytes(value: torch.Tensor) -> int:
    return int(value.numel() * value.element_size())


def _rank_map(tokens: int, positions: torch.Tensor, device: torch.device) -> torch.Tensor:
    result = torch.full((tokens,), -1, dtype=torch.int32, device=device)
    if positions.numel():
        result[positions.long()] = torch.arange(
            positions.numel(), dtype=torch.int32, device=device
        )
    return result


def _normalize_indices(
    indices: torch.Tensor,
    *,
    batch: int,
    heads: int,
    tokens: int,
    device: torch.device,
    validate_bounds: bool,
) -> torch.Tensor:
    if indices.ndim == 1:
        indices = indices[None, None, :].expand(batch, heads, -1)
    elif indices.ndim == 4 and indices.shape[-2] == 1:
        indices = indices.squeeze(-2)
    if indices.ndim != 3 or tuple(indices.shape[:2]) != (batch, heads):
        raise ValueError("indices must have shape [selected] or [batch, heads, selected]")
    logical = indices.to(device=device, dtype=torch.int32).contiguous()
    if validate_bounds and logical.numel() and (
        int(logical.min().item()) < 0 or int(logical.max().item()) >= tokens
    ):
        raise IndexError("cache index is outside the logical sequence")
    return logical


class BF16CrossLoopKVLayerBuilder:
    """Incrementally build a BF16 base-plus-sparse-override cache layer."""

    def __init__(self) -> None:
        self._loop1_key: torch.Tensor | None = None
        self._loop1_value: torch.Tensor | None = None
        self._loop2_key: torch.Tensor | None = None
        self._loop2_value: torch.Tensor | None = None

    @classmethod
    def from_loop1(
        cls,
        loop1_key: torch.Tensor,
        loop1_value: torch.Tensor,
        **_: object,
    ) -> "BF16CrossLoopKVLayerBuilder":
        if loop1_key.ndim != 4 or loop1_key.shape[0] != 1:
            raise ValueError("Loop-1 K/V must match [1, heads, tokens, channels]")
        if loop1_value.shape != loop1_key.shape:
            raise ValueError("Loop-1 K/V shapes must match")
        if loop1_value.dtype != loop1_key.dtype or loop1_value.device != loop1_key.device:
            raise ValueError("Loop-1 K/V must share dtype and device")
        result = cls()
        result._loop1_key = loop1_key.clone()
        result._loop1_value = loop1_value.clone()
        return result

    @property
    def retains_dense_loop1_prefix(self) -> bool:
        return True

    def add_loop2(self, loop2_key: torch.Tensor, loop2_value: torch.Tensor) -> None:
        if self._loop1_key is None or self._loop1_value is None:
            raise RuntimeError("Loop 1 is not installed")
        if self._loop2_key is not None:
            raise RuntimeError("Loop 2 is already installed")
        if loop2_key.shape != self._loop1_key.shape or loop2_value.shape != loop2_key.shape:
            raise ValueError("Loop-2 K/V shape differs from Loop 1")
        if loop2_key.dtype != self._loop1_key.dtype or loop2_key.device != self._loop1_key.device:
            raise ValueError("Loop-2 K/V dtype or device differs from Loop 1")
        self._loop2_key = loop2_key.clone()
        self._loop2_value = loop2_value.clone()

    def finalize(
        self,
        *,
        loop3_positions: torch.Tensor,
        loop3_keys: torch.Tensor,
        loop3_values: torch.Tensor,
        loop4_positions: torch.Tensor,
        loop4_keys: torch.Tensor,
        loop4_values: torch.Tensor,
    ) -> "BF16CrossLoopKVLayer":
        if any(value is None for value in (
            self._loop1_key, self._loop1_value, self._loop2_key, self._loop2_value
        )):
            raise RuntimeError("Loop 1/2 must be installed before finalization")
        return BF16CrossLoopKVLayer.from_prefill_updates(
            loop1_key=self._loop1_key,  # type: ignore[arg-type]
            loop1_value=self._loop1_value,  # type: ignore[arg-type]
            loop2_key=self._loop2_key,  # type: ignore[arg-type]
            loop2_value=self._loop2_value,  # type: ignore[arg-type]
            loop3_positions=loop3_positions,
            loop3_keys=loop3_keys,
            loop3_values=loop3_values,
            loop4_positions=loop4_positions,
            loop4_keys=loop4_keys,
            loop4_values=loop4_values,
        )


class BF16CrossLoopKVLayer:
    """Two dense BF16 anchors plus nested BF16 late-loop overrides."""

    storage_kind = "bf16_cross_loop"
    reader_backend = "bf16_cross_loop"

    def __init__(self) -> None:
        self._base_keys: tuple[torch.Tensor, torch.Tensor]
        self._base_values: tuple[torch.Tensor, torch.Tensor]
        self._late_keys: tuple[torch.Tensor, torch.Tensor]
        self._late_values: tuple[torch.Tensor, torch.Tensor]
        self.override_indices: tuple[torch.Tensor, torch.Tensor]
        self.rank_maps: tuple[torch.Tensor, torch.Tensor]
        self._decode_keys: list[torch.Tensor]
        self._decode_values: list[torch.Tensor]
        self._prompt_tokens = 0
        self._shape_prefix = (0, 0, 0)
        self._dtype = torch.bfloat16
        self._device = torch.device("cpu")

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
    ) -> "BF16CrossLoopKVLayer":
        dense = (loop1_key, loop1_value, loop2_key, loop2_value)
        if loop1_key.ndim != 4 or loop1_key.shape[0] != 1:
            raise ValueError("dense prefill K/V must have shape [1, heads, tokens, channels]")
        if any(tensor.shape != loop1_key.shape for tensor in dense):
            raise ValueError("Loop-1/2 K/V tensors must share a shape")
        if any(tensor.dtype != loop1_key.dtype or tensor.device != loop1_key.device for tensor in dense):
            raise ValueError("Loop-1/2 K/V tensors must share dtype and device")
        batch, heads, tokens, channels = map(int, loop1_key.shape)
        positions = []
        for logical, key, value in (
            (loop3_positions, loop3_keys, loop3_values),
            (loop4_positions, loop4_keys, loop4_values),
        ):
            logical = logical.to(device=loop1_key.device, dtype=torch.int32).contiguous()
            if logical.ndim != 1 or (
                logical.numel() > 1 and not bool(torch.all(logical[1:] > logical[:-1]).item())
            ):
                raise ValueError("late-loop positions must be strictly sorted")
            if logical.numel() and (
                int(logical.min().item()) < 0 or int(logical.max().item()) >= tokens
            ):
                raise IndexError("late-loop update is outside the prompt")
            expected = (batch, heads, int(logical.numel()), channels)
            if tuple(key.shape) != expected or value.shape != key.shape:
                raise ValueError("late-loop compact K/V shape differs from its positions")
            if key.dtype != loop1_key.dtype or key.device != loop1_key.device:
                raise ValueError("late-loop K/V dtype or device differs from Loop 1")
            positions.append(logical)
        positions3, positions4 = positions
        if positions4.numel():
            ranks4 = torch.searchsorted(positions3, positions4)
            valid = ranks4 < positions3.numel()
            matched = torch.zeros_like(valid)
            matched[valid] = positions3[ranks4[valid]] == positions4[valid]
            if not bool(matched.all().item()):
                raise ValueError("Loop-4 positions must be nested in Loop-3 positions")

        result = cls()
        result._prompt_tokens = tokens
        result._shape_prefix = (batch, heads, channels)
        result._dtype = loop1_key.dtype
        result._device = loop1_key.device
        result._base_keys = (loop1_key.clone(), loop2_key.clone())
        result._base_values = (loop1_value.clone(), loop2_value.clone())
        result._late_keys = (loop3_keys.clone(), loop4_keys.clone())
        result._late_values = (loop3_values.clone(), loop4_values.clone())
        result.override_indices = (positions3, positions4)
        result.rank_maps = (
            _rank_map(tokens, positions3, result._device),
            _rank_map(tokens, positions4, result._device),
        )
        empty = torch.empty(
            (batch, heads, 0, channels), dtype=result._dtype, device=result._device
        )
        result._decode_keys = [empty.clone() for _ in range(4)]
        result._decode_values = [empty.clone() for _ in range(4)]
        return result

    @property
    def token_count(self) -> int:
        return self._prompt_tokens + int(self._decode_keys[0].shape[2])

    @property
    def tail_tokens(self) -> int:
        return int(self._decode_keys[0].shape[2])

    @property
    def quantized_tokens(self) -> int:
        return 0

    @property
    def storage_bytes(self) -> int:
        tensors = (
            *self._base_keys,
            *self._base_values,
            *self._late_keys,
            *self._late_values,
            *self._decode_keys,
            *self._decode_values,
            *self.override_indices,
            *self.rank_maps,
        )
        return sum(_tensor_bytes(tensor) for tensor in tensors)

    @property
    def dense_bf16_bytes(self) -> int:
        batch, heads, channels = self._shape_prefix
        element_size = torch.empty((), dtype=self._dtype).element_size()
        return 4 * 2 * batch * heads * self.token_count * channels * element_size

    @property
    def storage_ratio(self) -> float:
        return self.storage_bytes / self.dense_bf16_bytes

    def _selected(self, loop: int, logical: torch.Tensor, *, values: bool) -> torch.Tensor:
        batch, heads, channels = self._shape_prefix
        prompt = logical < self._prompt_tokens
        safe_prompt = torch.where(prompt, logical, torch.zeros_like(logical)).long()
        bases = self._base_values if values else self._base_keys
        late = self._late_values if values else self._late_keys
        base = bases[0 if loop == 0 else 1]
        selected = torch.gather(
            base,
            2,
            safe_prompt.unsqueeze(-1).expand(batch, heads, -1, channels),
        )
        for stream in range(max(0, loop - 1)):
            ranks = self.rank_maps[stream].index_select(0, safe_prompt.flatten()).view_as(logical)
            active = prompt & (ranks >= 0)
            safe_rank = torch.where(active, ranks, torch.zeros_like(ranks)).long()
            update = torch.gather(
                late[stream],
                2,
                safe_rank.unsqueeze(-1).expand(batch, heads, -1, channels),
            )
            selected = torch.where(active.unsqueeze(-1), update, selected)
        if self.tail_tokens:
            tail = ~prompt
            safe_tail = torch.where(
                tail, logical - self._prompt_tokens, torch.zeros_like(logical)
            ).long()
            decode = self._decode_values[loop] if values else self._decode_keys[loop]
            tail_selected = torch.gather(
                decode,
                2,
                safe_tail.unsqueeze(-1).expand(batch, heads, -1, channels),
            )
            selected = torch.where(tail.unsqueeze(-1), tail_selected, selected)
        return selected

    def gather(self, loop: int, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        loop = int(loop)
        if loop not in range(4):
            raise ValueError("loop must be in [0, 3]")
        if indices.ndim != 1:
            raise ValueError("indices must be one-dimensional")
        logical = _normalize_indices(
            indices,
            batch=self._shape_prefix[0],
            heads=self._shape_prefix[1],
            tokens=self.token_count,
            device=self._device,
            validate_bounds=True,
        )
        return self._selected(loop, logical, values=False), self._selected(loop, logical, values=True)

    def qk(
        self,
        loop: int,
        query: torch.Tensor,
        indices: torch.Tensor,
        *,
        validate_indices: bool = True,
    ) -> torch.Tensor:
        batch, heads, channels = self._shape_prefix
        if tuple(query.shape) != (batch, heads, 1, channels):
            raise ValueError("query shape differs from this cache layer")
        logical = _normalize_indices(
            indices,
            batch=batch,
            heads=heads,
            tokens=self.token_count,
            device=self._device,
            validate_bounds=validate_indices,
        )
        selected = self._selected(int(loop), logical, values=False)
        return query.float() @ selected.float().transpose(-1, -2)

    def pv(
        self,
        loop: int,
        weights: torch.Tensor,
        indices: torch.Tensor,
        *,
        validate_indices: bool = True,
    ) -> torch.Tensor:
        batch, heads, _ = self._shape_prefix
        logical = _normalize_indices(
            indices,
            batch=batch,
            heads=heads,
            tokens=self.token_count,
            device=self._device,
            validate_bounds=validate_indices,
        )
        if tuple(weights.shape) != (batch, heads, 1, logical.shape[-1]):
            raise ValueError("weights must match [batch, heads, 1, selected]")
        selected = self._selected(int(loop), logical, values=True)
        return weights.float() @ selected.float()

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
        logical = _normalize_indices(
            indices,
            batch=self._shape_prefix[0],
            heads=self._shape_prefix[1],
            tokens=max(self.token_count, int(current_position) + 1),
            device=self._device,
            validate_bounds=False,
        )
        valid = valid.to(device=self._device, dtype=torch.bool)
        is_current = valid & (logical == int(current_position))
        safe = torch.where(is_current, torch.zeros_like(logical), logical)
        logits = self.qk(loop, query, safe, validate_indices=False)
        current_logits = query.float() @ current_key.float().transpose(-1, -2)
        logits = torch.where(is_current[:, :, None, :], current_logits, logits)
        logits = (logits * float(scaling)).masked_fill(~valid[:, :, None, :], float("-inf"))
        weights = torch.softmax(logits, dim=-1, dtype=torch.float32)
        target = self.pv(
            loop,
            weights * (~is_current)[:, :, None, :],
            safe,
            validate_indices=False,
        )
        target += torch.sum(weights * is_current[:, :, None, :], dim=-1, keepdim=True) * current_value.float()
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
        logical = torch.arange(self.token_count, device=self._device, dtype=torch.int32)
        logical = logical[None, None].expand(self._shape_prefix[0], self._shape_prefix[1], -1)
        past_logits = self.qk(loop, query, logical, validate_indices=False)
        current_logits = query.float() @ current_key.float().transpose(-1, -2)
        logits = torch.cat((past_logits, current_logits), dim=-1) * float(scaling)
        weights = torch.softmax(logits, dim=-1, dtype=torch.float32)
        output = self.pv(loop, weights[..., :-1], logical, validate_indices=False)
        output += weights[..., -1:] * current_value.float()
        return output, logits if return_logits else None

    def append_decode_token(
        self,
        keys: Sequence[torch.Tensor],
        values: Sequence[torch.Tensor],
    ) -> None:
        if len(keys) != 4 or len(values) != 4:
            raise ValueError("exactly four loop K/V tensors are required")
        expected = (*self._shape_prefix[:2], 1, self._shape_prefix[2])
        for index, (key, value) in enumerate(zip(keys, values, strict=True)):
            if tuple(key.shape) != expected or value.shape != key.shape:
                raise ValueError("decode K/V shape does not match the cache")
            self._decode_keys[index] = torch.cat((self._decode_keys[index], key), dim=2)
            self._decode_values[index] = torch.cat((self._decode_values[index], value), dim=2)
