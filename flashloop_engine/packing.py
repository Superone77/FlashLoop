"""Physical asymmetric 4-bit packing used by the FlashLoop cache.

The Torch implementation is the correctness reference for the Triton kernels. Codes
are always packed along the channel dimension: the even channel occupies the low
nibble and the odd channel occupies the high nibble.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


_FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _validate_values(values: torch.Tensor, group_size: int) -> int:
    if values.ndim != 4:
        raise ValueError("KV values must have shape [batch, heads, tokens, channels]")
    if values.dtype not in _FLOAT_DTYPES:
        raise TypeError("KV values must use a floating-point dtype")
    if values.shape[-2] <= 0 or values.shape[-1] <= 0:
        raise ValueError("KV values require non-empty token and channel dimensions")
    group_size = int(group_size)
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    return group_size


def _codes_for_group(
    values: torch.Tensor,
    *,
    reduce_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    work = values.float()
    minimum = work.amin(dim=reduce_dim, keepdim=True)
    maximum = work.amax(dim=reduce_dim, keepdim=True)
    span = maximum - minimum
    scale = torch.where(span > 0, span / 15.0, torch.ones_like(span))
    codes = torch.round((work - minimum) / scale).clamp_(0, 15).to(torch.uint8)
    return codes, minimum.to(torch.float16), scale.to(torch.float16)


def _pack_channel_nibbles(codes: torch.Tensor) -> torch.Tensor:
    low = codes[..., 0::2]
    high = codes[..., 1::2]
    if high.shape[-1] != low.shape[-1]:
        high = torch.cat([high, torch.zeros_like(low[..., :1])], dim=-1)
    return torch.bitwise_or(low, torch.bitwise_left_shift(high, 4)).contiguous()


def _unpack_channel_nibbles(payload: torch.Tensor, channels: int) -> torch.Tensor:
    low = torch.bitwise_and(payload, 0x0F)
    high = torch.bitwise_right_shift(payload, 4)
    codes = torch.empty(
        (*payload.shape[:-1], int(channels)),
        dtype=torch.uint8,
        device=payload.device,
    )
    codes[..., 0::2] = low
    if channels > 1:
        codes[..., 1::2] = high[..., : channels // 2]
    return codes


@dataclass(frozen=True)
class Packed4Bit:
    """Packed payload and two-byte affine metadata."""

    payload: torch.Tensor
    minimum: torch.Tensor
    scale: torch.Tensor
    original_shape: tuple[int, int, int, int]
    original_dtype: torch.dtype
    axis: str
    group_size: int
    row_groups: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.payload.dtype != torch.uint8:
            raise TypeError("packed payload must be uint8")
        if self.minimum.dtype != torch.float16 or self.scale.dtype != torch.float16:
            raise TypeError("packed metadata must be float16")
        if self.minimum.shape != self.scale.shape:
            raise ValueError("minimum and scale shapes differ")
        if self.axis not in {"per_channel", "per_token"}:
            raise ValueError("unknown packed axis")
        if self.row_groups is not None:
            if self.axis != "per_channel":
                raise ValueError("explicit row groups are only valid for per-channel K")
            if self.row_groups.dtype != torch.int32 or self.row_groups.ndim != 1:
                raise TypeError("row_groups must be one-dimensional int32")
            if self.row_groups.numel() != self.original_shape[2]:
                raise ValueError("row_groups must identify every stored token row")

    @property
    def storage_bytes(self) -> int:
        tensors = [self.payload, self.minimum, self.scale]
        if self.row_groups is not None:
            tensors.append(self.row_groups)
        return sum(_tensor_bytes(tensor) for tensor in tensors)

    def decode(self) -> torch.Tensor:
        batch, heads, tokens, channels = self.original_shape
        if tuple(self.payload.shape[:3]) != (batch, heads, tokens):
            raise RuntimeError("packed payload shape no longer matches metadata")
        codes = _unpack_channel_nibbles(self.payload, channels).float()
        if self.axis == "per_channel":
            if self.row_groups is None:
                minimum = self.minimum.repeat_interleave(self.group_size, dim=2)[
                    :, :, :tokens, :
                ]
                scale = self.scale.repeat_interleave(self.group_size, dim=2)[
                    :, :, :tokens, :
                ]
            else:
                groups = self.row_groups.to(device=self.minimum.device, dtype=torch.long)
                minimum = self.minimum.index_select(2, groups)
                scale = self.scale.index_select(2, groups)
        else:
            minimum = self.minimum.repeat_interleave(self.group_size, dim=3)[
                :, :, :, :channels
            ]
            scale = self.scale.repeat_interleave(self.group_size, dim=3)[
                :, :, :, :channels
            ]
        return (codes * scale.float() + minimum.float()).to(self.original_dtype)

    def decode_rows(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim != 1:
            raise ValueError("Torch reference decode_rows expects one-dimensional indices")
        indices = indices.to(device=self.payload.device, dtype=torch.long)
        if indices.numel() and (
            int(indices.min().item()) < 0
            or int(indices.max().item()) >= self.original_shape[2]
        ):
            raise IndexError("decode row index is outside the packed token range")
        return self.decode().index_select(2, indices)


def pack_k_per_channel(values: torch.Tensor, *, group_size: int = 64) -> Packed4Bit:
    """Pack K with affine parameters per channel over token groups."""

    group_size = _validate_values(values, group_size)
    code_chunks: list[torch.Tensor] = []
    minima: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    for start in range(0, int(values.shape[2]), group_size):
        codes, minimum, scale = _codes_for_group(
            values[:, :, start : start + group_size, :],
            reduce_dim=2,
        )
        code_chunks.append(codes)
        minima.append(minimum)
        scales.append(scale)
    all_codes = torch.cat(code_chunks, dim=2)
    return Packed4Bit(
        payload=_pack_channel_nibbles(all_codes),
        minimum=torch.cat(minima, dim=2).contiguous(),
        scale=torch.cat(scales, dim=2).contiguous(),
        original_shape=tuple(int(value) for value in values.shape),
        original_dtype=values.dtype,
        axis="per_channel",
        group_size=group_size,
    )


def pack_v_per_token(values: torch.Tensor, *, group_size: int = 64) -> Packed4Bit:
    """Pack V with affine parameters per token over channel groups."""

    group_size = _validate_values(values, group_size)
    code_chunks: list[torch.Tensor] = []
    minima: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    for start in range(0, int(values.shape[3]), group_size):
        codes, minimum, scale = _codes_for_group(
            values[..., start : start + group_size],
            reduce_dim=3,
        )
        code_chunks.append(codes)
        minima.append(minimum)
        scales.append(scale)
    all_codes = torch.cat(code_chunks, dim=3)
    return Packed4Bit(
        payload=_pack_channel_nibbles(all_codes),
        minimum=torch.cat(minima, dim=3).contiguous(),
        scale=torch.cat(scales, dim=3).contiguous(),
        original_shape=tuple(int(value) for value in values.shape),
        original_dtype=values.dtype,
        axis="per_token",
        group_size=group_size,
    )


def pack_k_per_channel_grouped(
    values: torch.Tensor,
    *,
    logical_groups: torch.Tensor,
) -> Packed4Bit:
    """Pack sparse K rows without merging rows from different logical groups."""

    _validate_values(values, 1)
    if logical_groups.ndim != 1 or logical_groups.numel() != values.shape[2]:
        raise ValueError("logical_groups must identify every sparse token row")
    if logical_groups.dtype not in (torch.int32, torch.int64):
        raise TypeError("logical_groups must use an integer dtype")
    logical_groups = logical_groups.to(device=values.device, dtype=torch.long)
    if logical_groups.numel() == 0 or int(logical_groups.min().item()) < 0:
        raise ValueError("logical_groups must be non-empty and non-negative")
    if logical_groups.numel() > 1 and not bool(
        torch.all(logical_groups[1:] >= logical_groups[:-1]).item()
    ):
        raise ValueError("logical_groups must be sorted")

    _unique, inverse = torch.unique_consecutive(
        logical_groups,
        return_inverse=True,
    )
    code_chunks: list[torch.Tensor] = []
    minima: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    for compact_group in range(int(inverse.max().item()) + 1):
        rows = values[:, :, inverse == compact_group, :]
        codes, minimum, scale = _codes_for_group(rows, reduce_dim=2)
        code_chunks.append(codes)
        minima.append(minimum)
        scales.append(scale)
    codes = torch.cat(code_chunks, dim=2)
    return Packed4Bit(
        payload=_pack_channel_nibbles(codes),
        minimum=torch.cat(minima, dim=2).contiguous(),
        scale=torch.cat(scales, dim=2).contiguous(),
        original_shape=tuple(int(value) for value in values.shape),
        original_dtype=values.dtype,
        axis="per_channel",
        group_size=1,
        row_groups=inverse.to(torch.int32).contiguous(),
    )
