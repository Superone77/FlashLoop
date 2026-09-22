"""KIVI-style output-parallel physical 4-bit KV storage.

KIVI's CUDA GEMV assigns warps to output features, so its physical payload packs
eight adjacent output values into each int32 word.  For K, tokens are output
features; for V, channels are output features.  This module owns the layout and
its Torch correctness reference.  CUDA readers are added separately so the
physical-byte accounting never relies on a hidden row-major cache copy.

The layout follows the MIT-licensed KIVI implementation:
https://github.com/jy-yuan/KIVI
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
import importlib
from typing import Any

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on minimal CPU installs.
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


_FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}
_PACK_FACTOR = 8
_EXTENSION: Any | None = None


def load_kivi_outer_extension() -> Any:
    """Load the separately compiled KIVI-derived CUDA reader once."""

    global _EXTENSION
    if _EXTENSION is None:
        try:
            _EXTENSION = importlib.import_module("flashloop_kivi_gemv")
        except ImportError as error:
            raise RuntimeError(
                "the kivi_outer backend requires the flashloop_kivi_gemv CUDA extension"
            ) from error
    return _EXTENSION


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


if _TRITON_AVAILABLE:

    @triton.jit
    def _round_nearest_even_int32(value):
        return tl.inline_asm_elementwise(
            "cvt.rni.s32.f32 $0, $1;",
            "=r,f",
            [value],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _subtract_divide_round_nearest(value, minimum, scale):
        return tl.inline_asm_elementwise(
            "{ .reg .f32 numerator; sub.rn.f32 numerator, $1, $2; "
            "div.rn.f32 $0, numerator, $3; }",
            "=f,f,f,f",
            [value, minimum, scale],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _pack_kivi_output_words_kernel(
        values,
        work_minimum,
        work_scale,
        payload,
        total_words: tl.constexpr,
        tokens: tl.constexpr,
        channels: tl.constexpr,
        token_groups: tl.constexpr,
        channel_groups: tl.constexpr,
        group_size: tl.constexpr,
        PACK_FACTOR: tl.constexpr,
        PER_CHANNEL: tl.constexpr,
        BLOCK_WORDS: tl.constexpr,
    ):
        word = tl.program_id(0) * BLOCK_WORDS + tl.arange(0, BLOCK_WORDS)
        valid_word = word < total_words
        if PER_CHANNEL:
            packed_outputs = tl.cdiv(tokens, PACK_FACTOR)
            matrix_words = packed_outputs * channels
            batch_head = word // matrix_words
            within = word % matrix_words
            packed_output = within // channels
            channel = within % channels
            packed = tl.zeros((BLOCK_WORDS,), dtype=tl.int32)
            for nibble in range(PACK_FACTOR):
                token = packed_output * PACK_FACTOR + nibble
                valid = valid_word & (token < tokens)
                value_offset = (batch_head * tokens + token) * channels + channel
                metadata_offset = (
                    (batch_head * token_groups + token // group_size) * channels
                    + channel
                )
                value = tl.load(values + value_offset, mask=valid, other=0.0).to(tl.float32)
                minimum = tl.load(
                    work_minimum + metadata_offset, mask=valid, other=0.0
                )
                scale = tl.load(work_scale + metadata_offset, mask=valid, other=1.0)
                quantized = tl.maximum(
                    0.0,
                    tl.minimum(
                        15.0,
                        _subtract_divide_round_nearest(value, minimum, scale),
                    ),
                )
                code = _round_nearest_even_int32(quantized)
                packed |= code << (4 * nibble)
        else:
            packed_outputs = tl.cdiv(channels, PACK_FACTOR)
            matrix_words = packed_outputs * tokens
            batch_head = word // matrix_words
            within = word % matrix_words
            packed_output = within // tokens
            token = within % tokens
            packed = tl.zeros((BLOCK_WORDS,), dtype=tl.int32)
            for nibble in range(PACK_FACTOR):
                channel = packed_output * PACK_FACTOR + nibble
                valid = valid_word & (channel < channels)
                value_offset = (batch_head * tokens + token) * channels + channel
                metadata_offset = (
                    (batch_head * tokens + token) * channel_groups
                    + channel // group_size
                )
                value = tl.load(values + value_offset, mask=valid, other=0.0).to(tl.float32)
                minimum = tl.load(
                    work_minimum + metadata_offset, mask=valid, other=0.0
                )
                scale = tl.load(work_scale + metadata_offset, mask=valid, other=1.0)
                quantized = tl.maximum(
                    0.0,
                    tl.minimum(
                        15.0,
                        _subtract_divide_round_nearest(value, minimum, scale),
                    ),
                )
                code = _round_nearest_even_int32(quantized)
                packed |= code << (4 * nibble)
        tl.store(payload + word, packed, mask=valid_word)


def _triton_pack_kivi(
    values: torch.Tensor,
    *,
    axis: str,
    group_size: int,
) -> KiviPacked4Bit:
    """KIVI-compatible physical pack with partial-group masking."""

    if not values.is_cuda or not _TRITON_AVAILABLE:
        raise RuntimeError("Triton KIVI packing requires CUDA and Triton")
    if values.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("Triton KIVI packing requires FP16 or BF16 input")
    if int(values.shape[-1]) != 128 or int(group_size) != 64:
        raise ValueError("the production KIVI pack specializes head_dim=128, group_size=64")
    if axis not in {"per_channel", "per_token"}:
        raise ValueError("unknown KIVI quantization axis")
    from .kernels.quantization import _k_minmax_kernel, _v_minmax_kernel

    values = values.contiguous()
    batch, heads, tokens, channels = (int(value) for value in values.shape)
    token_groups = triton.cdiv(tokens, group_size)
    channel_groups = triton.cdiv(channels, group_size)
    metadata_shape = (
        (batch, heads, token_groups, channels)
        if axis == "per_channel"
        else (batch, heads, tokens, channel_groups)
    )
    minimum = torch.empty(metadata_shape, device=values.device, dtype=torch.float16)
    scale = torch.empty_like(minimum)
    work_minimum = torch.empty(metadata_shape, device=values.device, dtype=torch.float32)
    work_scale = torch.empty_like(work_minimum)
    metadata_count = minimum.numel()
    grid = (triton.cdiv(metadata_count, 128),)
    if axis == "per_channel":
        _k_minmax_kernel[grid](
            values,
            work_minimum,
            work_scale,
            minimum,
            scale,
            tokens=tokens,
            channels=channels,
            token_groups=token_groups,
            group_size=group_size,
            metadata_count=metadata_count,
            BLOCK_ROWS=128,
            num_warps=8,
        )
        payload_shape = (batch, heads, triton.cdiv(tokens, _PACK_FACTOR), channels)
    else:
        _v_minmax_kernel[grid](
            values,
            work_minimum,
            work_scale,
            minimum,
            scale,
            tokens=tokens,
            channels=channels,
            channel_groups=channel_groups,
            group_size=group_size,
            metadata_count=metadata_count,
            BLOCK_ROWS=128,
            num_warps=8,
        )
        payload_shape = (batch, heads, triton.cdiv(channels, _PACK_FACTOR), tokens)
    payload = torch.empty(payload_shape, device=values.device, dtype=torch.int32)
    total_words = payload.numel()
    _pack_kivi_output_words_kernel[(triton.cdiv(total_words, 256),)](
        values,
        work_minimum,
        work_scale,
        payload,
        total_words=total_words,
        tokens=tokens,
        channels=channels,
        token_groups=token_groups,
        channel_groups=channel_groups,
        group_size=group_size,
        PACK_FACTOR=_PACK_FACTOR,
        PER_CHANNEL=axis == "per_channel",
        BLOCK_WORDS=256,
        num_warps=4,
    )
    if axis == "per_token":
        minimum = minimum.permute(0, 1, 3, 2).contiguous()
        scale = scale.permute(0, 1, 3, 2).contiguous()
    return KiviPacked4Bit(
        payload=payload,
        minimum=minimum,
        scale=scale,
        original_shape=(batch, heads, tokens, channels),
        original_dtype=values.dtype,
        axis=axis,
        group_size=group_size,
    )


def _validate(values: torch.Tensor, group_size: int) -> int:
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
    codes = torch.round((work - minimum) / scale).clamp_(0, 15).to(torch.int32)
    return codes, minimum.to(torch.float16), scale.to(torch.float16)


def _pack_output_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack matrix output rows in groups of eight into int32 words."""

    if codes.ndim != 4:
        raise ValueError("output codes must have shape [batch, heads, output, input]")
    output = int(codes.shape[2])
    padded = ((output + _PACK_FACTOR - 1) // _PACK_FACTOR) * _PACK_FACTOR
    if padded != output:
        codes = torch.cat(
            (
                codes,
                torch.zeros(
                    (*codes.shape[:2], padded - output, codes.shape[3]),
                    device=codes.device,
                    dtype=codes.dtype,
                ),
            ),
            dim=2,
        )
    words = codes.reshape(*codes.shape[:2], padded // _PACK_FACTOR, _PACK_FACTOR, codes.shape[3])
    shifts = (
        torch.arange(_PACK_FACTOR, device=codes.device, dtype=torch.int64)
        .mul_(4)
        .view(1, 1, 1, _PACK_FACTOR, 1)
    )
    return torch.sum(words.to(torch.int64) << shifts, dim=3).to(torch.int32).contiguous()


def _unpack_output_codes(payload: torch.Tensor, output: int) -> torch.Tensor:
    shifts = (
        torch.arange(_PACK_FACTOR, device=payload.device, dtype=torch.int64)
        .mul_(4)
        .view(1, 1, 1, _PACK_FACTOR, 1)
    )
    codes = (payload.to(torch.int64).unsqueeze(3) >> shifts) & 0x0F
    decoded = codes.reshape(*payload.shape[:2], payload.shape[2] * _PACK_FACTOR, payload.shape[3])
    return decoded[:, :, : int(output), :].float()


@dataclass(frozen=True)
class KiviPacked4Bit:
    """One physical KIVI-layout affine 4-bit matrix."""

    payload: torch.Tensor
    minimum: torch.Tensor
    scale: torch.Tensor
    original_shape: tuple[int, int, int, int]
    original_dtype: torch.dtype
    axis: str
    group_size: int

    def __post_init__(self) -> None:
        if self.payload.dtype != torch.int32:
            raise TypeError("KIVI payload must be int32")
        if self.minimum.dtype != torch.float16 or self.scale.dtype != torch.float16:
            raise TypeError("KIVI affine metadata must be float16")
        if self.minimum.shape != self.scale.shape:
            raise ValueError("minimum and scale shapes differ")
        if self.axis not in {"per_channel", "per_token"}:
            raise ValueError("unknown KIVI quantization axis")
        batch, heads, tokens, channels = self.original_shape
        if self.axis == "per_channel":
            expected_payload = (batch, heads, (tokens + 7) // 8, channels)
            expected_metadata = (
                batch,
                heads,
                (tokens + self.group_size - 1) // self.group_size,
                channels,
            )
        else:
            expected_payload = (batch, heads, (channels + 7) // 8, tokens)
            expected_metadata = (
                batch,
                heads,
                (channels + self.group_size - 1) // self.group_size,
                tokens,
            )
        if tuple(self.payload.shape) != expected_payload:
            raise ValueError("KIVI payload shape differs from logical KV shape")
        if tuple(self.minimum.shape) != expected_metadata:
            raise ValueError("KIVI metadata shape differs from logical KV shape")

    @property
    def storage_bytes(self) -> int:
        return sum(_tensor_bytes(tensor) for tensor in (self.payload, self.minimum, self.scale))

    @property
    def matrix_shape(self) -> tuple[int, int, int, int]:
        batch, heads, tokens, channels = self.original_shape
        return (
            (batch, heads, tokens, channels)
            if self.axis == "per_channel"
            else (batch, heads, channels, tokens)
        )

    def decode(self) -> torch.Tensor:
        batch, heads, tokens, channels = self.original_shape
        if self.axis == "per_channel":
            codes = _unpack_output_codes(self.payload, tokens)
            minimum = self.minimum.repeat_interleave(self.group_size, dim=2)[
                :, :, :tokens, :
            ]
            scale = self.scale.repeat_interleave(self.group_size, dim=2)[
                :, :, :tokens, :
            ]
            decoded = codes * scale.float() + minimum.float()
        else:
            codes = _unpack_output_codes(self.payload, channels)
            minimum = self.minimum.repeat_interleave(self.group_size, dim=2)[
                :, :, :channels, :
            ]
            scale = self.scale.repeat_interleave(self.group_size, dim=2)[
                :, :, :channels, :
            ]
            decoded = (codes * scale.float() + minimum.float()).transpose(2, 3)
        return decoded.to(self.original_dtype)


def pack_kivi_k_per_channel(
    values: torch.Tensor,
    *,
    group_size: int = 64,
) -> KiviPacked4Bit:
    """Pack K so one GEMV output feature corresponds to one token."""

    group_size = _validate(values, group_size)
    if values.is_cuda and _TRITON_AVAILABLE:
        return _triton_pack_kivi(
            values,
            axis="per_channel",
            group_size=group_size,
        )
    codes: list[torch.Tensor] = []
    minima: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    for start in range(0, int(values.shape[2]), group_size):
        code, minimum, scale = _codes_for_group(
            values[:, :, start : start + group_size, :],
            reduce_dim=2,
        )
        codes.append(code)
        minima.append(minimum)
        scales.append(scale)
    all_codes = torch.cat(codes, dim=2)
    return KiviPacked4Bit(
        payload=_pack_output_codes(all_codes),
        minimum=torch.cat(minima, dim=2).contiguous(),
        scale=torch.cat(scales, dim=2).contiguous(),
        original_shape=tuple(int(value) for value in values.shape),
        original_dtype=values.dtype,
        axis="per_channel",
        group_size=group_size,
    )


def pack_kivi_v_per_token(
    values: torch.Tensor,
    *,
    group_size: int = 64,
) -> KiviPacked4Bit:
    """Pack V so one GEMV output feature corresponds to one channel."""

    group_size = _validate(values, group_size)
    if values.is_cuda and _TRITON_AVAILABLE:
        return _triton_pack_kivi(
            values,
            axis="per_token",
            group_size=group_size,
        )
    codes: list[torch.Tensor] = []
    minima: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    for start in range(0, int(values.shape[3]), group_size):
        code, minimum, scale = _codes_for_group(
            values[..., start : start + group_size],
            reduce_dim=3,
        )
        codes.append(code)
        minima.append(minimum)
        scales.append(scale)
    matrix_codes = torch.cat(codes, dim=3).transpose(2, 3).contiguous()
    return KiviPacked4Bit(
        payload=_pack_output_codes(matrix_codes),
        minimum=torch.cat(minima, dim=3).permute(0, 1, 3, 2).contiguous(),
        scale=torch.cat(scales, dim=3).permute(0, 1, 3, 2).contiguous(),
        original_shape=tuple(int(value) for value in values.shape),
        original_dtype=values.dtype,
        axis="per_token",
        group_size=group_size,
    )


def outer_matmul_reference(inputs: torch.Tensor, packed: KiviPacked4Bit) -> torch.Tensor:
    """Reference GEMV using only the logical tensor reconstructed from payload."""

    if inputs.ndim != 4 or tuple(inputs.shape[:2]) != packed.original_shape[:2]:
        raise ValueError("input batch/head shape differs from packed storage")
    if packed.axis == "per_channel":
        expected = packed.original_shape[3]
        if int(inputs.shape[-1]) != expected:
            raise ValueError("input width differs from K channels")
        return torch.matmul(inputs.float(), packed.decode().float().transpose(-1, -2))
    expected = packed.original_shape[2]
    if int(inputs.shape[-1]) != expected:
        raise ValueError("input width differs from V tokens")
    return torch.matmul(inputs.float(), packed.decode().float())


def kivi_outer_gemv(
    inputs: torch.Tensor,
    packed: KiviPacked4Bit,
    *,
    extension: Any,
    require_cuda: bool = True,
) -> torch.Tensor:
    """Invoke FlashLoop's logical-size adaptation of KIVI's CUDA GEMV.

    The upstream kernel infers the output width from complete quantization
    groups.  FlashLoop's nested updates can contain any number of rows, so the
    adapted entry point receives the logical output size explicitly and masks
    the final packed word.  FP16 conversion is part of this measured adapter
    because upstream KIVI's public CUDA kernel operates on ``half``.
    """

    if inputs.ndim != 4 or tuple(inputs.shape[:2]) != packed.original_shape[:2]:
        raise ValueError("input batch/head shape differs from packed storage")
    if require_cuda and (not inputs.is_cuda or not packed.payload.is_cuda):
        raise RuntimeError("KIVI outer GEMV requires CUDA tensors")
    batch, heads = (int(value) for value in inputs.shape[:2])
    matrix_output, matrix_input = packed.matrix_shape[2:]
    if int(inputs.shape[-1]) != matrix_input:
        raise ValueError("input width differs from packed matrix")
    prepared_input = inputs.to(torch.float16).reshape(
        batch * heads,
        int(inputs.shape[2]),
        matrix_input,
    ).contiguous()
    payload = packed.payload.reshape(
        batch * heads,
        int(packed.payload.shape[2]),
        matrix_input,
    ).contiguous()
    scale = packed.scale.reshape(
        batch * heads,
        int(packed.scale.shape[2]),
        matrix_input,
    ).contiguous()
    minimum = packed.minimum.reshape_as(scale).contiguous()
    output = extension.gemv_forward_cuda_outer_dim_logical(
        prepared_input,
        payload,
        scale,
        minimum,
        4,
        packed.group_size,
        heads,
        heads,
        matrix_output,
    )
    expected = (batch * heads, int(inputs.shape[2]), matrix_output)
    if tuple(output.shape) != expected:
        raise RuntimeError("KIVI extension returned an unexpected output shape")
    return output.reshape(batch, heads, int(inputs.shape[2]), matrix_output).float()


def kivi_selected_qk(
    query: torch.Tensor,
    packed: KiviPacked4Bit,
    indices: torch.Tensor,
    *,
    extension: Any,
) -> torch.Tensor:
    """Read only selected K rows from the output-packed KIVI payload."""

    if packed.axis != "per_channel":
        raise ValueError("selected QK requires per-channel K storage")
    batch, heads, tokens, channels = packed.original_shape
    if tuple(query.shape) != (batch, heads, 1, channels):
        raise ValueError("query shape differs from K storage")
    if indices.ndim == 1:
        indices = indices[None, None].expand(batch, heads, -1)
    if indices.ndim != 3 or tuple(indices.shape[:2]) != (batch, heads):
        raise ValueError("selected K indices must have shape [batch, heads, selected]")
    prepared_query = query.to(torch.float16).reshape(batch * heads, 1, channels).contiguous()
    prepared_indices = indices.to(device=query.device, dtype=torch.int32).reshape(
        batch * heads, -1
    ).contiguous()
    output = extension.qk_selected_cuda(
        prepared_query,
        packed.payload.reshape(batch * heads, packed.payload.shape[2], channels).contiguous(),
        packed.scale.reshape(batch * heads, packed.scale.shape[2], channels).contiguous(),
        packed.minimum.reshape(batch * heads, packed.minimum.shape[2], channels).contiguous(),
        prepared_indices,
        packed.group_size,
        heads,
        heads,
        tokens,
    )
    return output.reshape(batch, heads, 1, prepared_indices.shape[-1]).float()


def kivi_selected_pv(
    weights: torch.Tensor,
    packed: KiviPacked4Bit,
    indices: torch.Tensor,
    *,
    extension: Any,
) -> torch.Tensor:
    """Accumulate only selected V rows from the input dimension of KIVI V."""

    if packed.axis != "per_token":
        raise ValueError("selected PV requires per-token V storage")
    batch, heads, tokens, channels = packed.original_shape
    if indices.ndim == 1:
        indices = indices[None, None].expand(batch, heads, -1)
    selected = int(indices.shape[-1])
    if tuple(weights.shape) != (batch, heads, 1, selected):
        raise ValueError("weights shape differs from selected V indices")
    prepared_weights = weights.to(torch.float16).reshape(batch * heads, 1, selected).contiguous()
    prepared_indices = indices.to(device=weights.device, dtype=torch.int32).reshape(
        batch * heads, selected
    ).contiguous()
    output = extension.pv_selected_cuda(
        prepared_weights,
        packed.payload.reshape(batch * heads, packed.payload.shape[2], tokens).contiguous(),
        packed.scale.reshape(batch * heads, packed.scale.shape[2], tokens).contiguous(),
        packed.minimum.reshape(batch * heads, packed.minimum.shape[2], tokens).contiguous(),
        prepared_indices,
        packed.group_size,
        heads,
        heads,
        tokens,
        channels,
    )
    return output.reshape(batch, heads, 1, channels).float()


def _validate_cross_loop_reader(
    streams: tuple[KiviPacked4Bit | None, ...],
    rank_maps: tuple[torch.Tensor, torch.Tensor],
    indices: torch.Tensor,
    *,
    loop: int,
    axis: str,
) -> tuple[KiviPacked4Bit, torch.Tensor]:
    if len(streams) != 4 or streams[0] is None:
        raise ValueError("four stream slots with a physical anchor are required")
    if len(rank_maps) != 2:
        raise ValueError("Loop-3/4 rank maps are required")
    if int(loop) not in range(4):
        raise ValueError("loop must be in [0, 3]")
    anchor = streams[0]
    assert anchor is not None
    if anchor.axis != axis:
        raise ValueError("cross-loop stream uses the wrong quantization axis")
    for stream in streams:
        if stream is not None and (
            stream.axis != axis
            or stream.original_shape[:2] != anchor.original_shape[:2]
            or stream.original_shape[3] != anchor.original_shape[3]
        ):
            raise ValueError("cross-loop streams have incompatible matrix shapes")
    if indices.ndim == 1:
        indices = indices[None, None].expand(*anchor.original_shape[:2], -1)
    elif indices.ndim == 4 and indices.shape[-2] == 1:
        indices = indices.squeeze(-2)
    if indices.ndim != 3 or tuple(indices.shape[:2]) != anchor.original_shape[:2]:
        raise ValueError("indices must have shape [selected] or [batch, heads, selected]")
    return anchor, indices.to(device=anchor.payload.device, dtype=torch.int64).contiguous()


def _prepare_fused_cross_loop_streams(
    streams: tuple[KiviPacked4Bit | None, ...],
    *,
    axis: str,
) -> tuple[KiviPacked4Bit, KiviPacked4Bit, KiviPacked4Bit, KiviPacked4Bit]:
    if len(streams) != 4 or any(stream is None for stream in streams):
        raise ValueError("the fused cross-loop reader requires four physical streams")
    prepared = tuple(streams)
    assert all(isinstance(stream, KiviPacked4Bit) for stream in prepared)
    anchor = prepared[0]
    if any(
        stream.axis != axis
        or stream.group_size != 64
        or stream.original_shape[:2] != anchor.original_shape[:2]
        or stream.original_shape[3] != anchor.original_shape[3]
        for stream in prepared
    ):
        raise ValueError("fused cross-loop streams have incompatible storage")
    return prepared  # type: ignore[return-value]


def kivi_fused_cross_loop_qk(
    query: torch.Tensor,
    streams: tuple[KiviPacked4Bit | None, ...],
    rank_maps: tuple[torch.Tensor, torch.Tensor],
    indices: torch.Tensor,
    *,
    loop: int,
    extension: Any,
    require_cuda: bool = True,
) -> torch.Tensor:
    """Read anchor plus all active K deltas in one CUDA launch."""

    prepared = _prepare_fused_cross_loop_streams(streams, axis="per_channel")
    anchor, logical = _validate_cross_loop_reader(
        streams, rank_maps, indices, loop=loop, axis="per_channel"
    )
    batch, heads, _, channels = anchor.original_shape
    if tuple(query.shape) != (batch, heads, 1, channels):
        raise ValueError("query shape differs from the K anchor")
    if require_cuda and (not query.is_cuda or not anchor.payload.is_cuda):
        raise RuntimeError("fused KIVI cross-loop QK requires CUDA tensors")
    prepared_query = query.to(torch.float16).reshape(batch * heads, 1, channels).contiguous()
    prepared_indices = logical.to(dtype=torch.int32).reshape(batch * heads, -1).contiguous()
    payloads = [
        stream.payload.reshape(batch * heads, stream.payload.shape[2], channels).contiguous()
        for stream in prepared
    ]
    scales = [
        stream.scale.reshape(batch * heads, stream.scale.shape[2], channels).contiguous()
        for stream in prepared
    ]
    minima = [
        stream.minimum.reshape(batch * heads, stream.minimum.shape[2], channels).contiguous()
        for stream in prepared
    ]
    output = extension.qk_cross_loop_cuda(
        prepared_query,
        payloads,
        scales,
        minima,
        prepared_indices,
        rank_maps[0].to(device=query.device, dtype=torch.int32).contiguous(),
        rank_maps[1].to(device=query.device, dtype=torch.int32).contiguous(),
        64,
        heads,
        heads,
        int(loop),
        tuple(int(stream.original_shape[2]) for stream in prepared),
    )
    return output.reshape(batch, heads, 1, prepared_indices.shape[-1]).float()


def kivi_fused_cross_loop_pv(
    weights: torch.Tensor,
    streams: tuple[KiviPacked4Bit | None, ...],
    rank_maps: tuple[torch.Tensor, torch.Tensor],
    indices: torch.Tensor,
    *,
    loop: int,
    extension: Any,
    require_cuda: bool = True,
) -> torch.Tensor:
    """Accumulate anchor plus all active V deltas in one CUDA launch."""

    prepared = _prepare_fused_cross_loop_streams(streams, axis="per_token")
    anchor, logical = _validate_cross_loop_reader(
        streams, rank_maps, indices, loop=loop, axis="per_token"
    )
    batch, heads, _, channels = anchor.original_shape
    selected = int(logical.shape[-1])
    if tuple(weights.shape) != (batch, heads, 1, selected):
        raise ValueError("weights shape differs from selected logical rows")
    if require_cuda and (not weights.is_cuda or not anchor.payload.is_cuda):
        raise RuntimeError("fused KIVI cross-loop PV requires CUDA tensors")
    prepared_weights = weights.to(torch.float16).reshape(batch * heads, 1, selected).contiguous()
    prepared_indices = logical.to(dtype=torch.int32).reshape(batch * heads, selected).contiguous()
    payloads = [
        stream.payload.reshape(batch * heads, stream.payload.shape[2], stream.original_shape[2]).contiguous()
        for stream in prepared
    ]
    scales = [
        stream.scale.reshape(batch * heads, stream.scale.shape[2], stream.original_shape[2]).contiguous()
        for stream in prepared
    ]
    minima = [
        stream.minimum.reshape(batch * heads, stream.minimum.shape[2], stream.original_shape[2]).contiguous()
        for stream in prepared
    ]
    output = extension.pv_cross_loop_cuda(
        prepared_weights,
        payloads,
        scales,
        minima,
        prepared_indices,
        rank_maps[0].to(device=weights.device, dtype=torch.int32).contiguous(),
        rank_maps[1].to(device=weights.device, dtype=torch.int32).contiguous(),
        64,
        heads,
        heads,
        int(loop),
        tuple(int(stream.original_shape[2]) for stream in prepared),
        channels,
    )
    return output.reshape(batch, heads, 1, channels).float()


def kivi_fused_sparse_attention_delta(
    *,
    query: torch.Tensor,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    key_streams: tuple[KiviPacked4Bit | None, ...],
    value_streams: tuple[KiviPacked4Bit | None, ...],
    rank_maps: tuple[torch.Tensor, torch.Tensor],
    tail_key: torch.Tensor,
    tail_value: torch.Tensor,
    indices: torch.Tensor,
    valid: torch.Tensor,
    source_selected_output: torch.Tensor,
    global_mass: torch.Tensor,
    current_position: int,
    scaling: float,
    loop: int,
    extension: Any,
    require_cuda: bool = True,
    return_current_logits: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Fuse KIVI QK, subset softmax, PV, and cached-mass correction."""

    keys = _prepare_fused_cross_loop_streams(key_streams, axis="per_channel")
    values = _prepare_fused_cross_loop_streams(value_streams, axis="per_token")
    anchor, logical = _validate_cross_loop_reader(
        key_streams, rank_maps, indices, loop=loop, axis="per_channel"
    )
    batch, heads, _, channels = anchor.original_shape
    expected_vector = (batch, heads, 1, channels)
    if any(
        tuple(tensor.shape) != expected_vector
        for tensor in (query, current_key, current_value, source_selected_output)
    ):
        raise ValueError("fused attention vectors differ from the KIVI anchor")
    if tuple(valid.shape) != tuple(logical.shape):
        raise ValueError("valid mask must match selected indices")
    if tuple(global_mass.shape) != (batch, heads, 1, 1):
        raise ValueError("global mass must have shape [batch, heads, 1, 1]")
    if tuple(tail_key.shape[:2]) != (batch, heads) or tuple(tail_value.shape) != tuple(
        tail_key.shape
    ) or int(tail_key.shape[-1]) != channels:
        raise ValueError("tail K/V shape differs from the KIVI anchor")
    if any(tensor.dtype != torch.bfloat16 for tensor in (current_key, current_value, tail_key, tail_value)):
        raise ValueError("fused attention currently specializes BF16 tail/current K/V")
    if query.dtype != torch.bfloat16:
        raise ValueError("fused attention currently specializes BF16 queries")
    if require_cuda and (
        not query.is_cuda
        or not current_key.is_cuda
        or not anchor.payload.is_cuda
    ):
        raise RuntimeError("fused KIVI sparse attention requires CUDA tensors")
    selected = int(logical.shape[-1])
    prepared_query = query.to(torch.float16).reshape(batch * heads, 1, channels).contiguous()
    prepared_indices = logical.to(dtype=torch.int32).reshape(batch * heads, selected).contiguous()
    key_payloads = [
        stream.payload.reshape(batch * heads, stream.payload.shape[2], channels).contiguous()
        for stream in keys
    ]
    key_scales = [
        stream.scale.reshape(batch * heads, stream.scale.shape[2], channels).contiguous()
        for stream in keys
    ]
    key_minima = [
        stream.minimum.reshape(batch * heads, stream.minimum.shape[2], channels).contiguous()
        for stream in keys
    ]
    value_payloads = [
        stream.payload.reshape(batch * heads, stream.payload.shape[2], stream.original_shape[2]).contiguous()
        for stream in values
    ]
    value_scales = [
        stream.scale.reshape(batch * heads, stream.scale.shape[2], stream.original_shape[2]).contiguous()
        for stream in values
    ]
    value_minima = [
        stream.minimum.reshape(batch * heads, stream.minimum.shape[2], stream.original_shape[2]).contiguous()
        for stream in values
    ]
    raw_outputs = extension.sparse_attention_delta_cuda(
        prepared_query,
        query.reshape(batch * heads, channels).contiguous(),
        current_key.reshape(batch * heads, channels).contiguous(),
        current_value.reshape(batch * heads, channels).contiguous(),
        key_payloads,
        key_scales,
        key_minima,
        value_payloads,
        value_scales,
        value_minima,
        prepared_indices,
        valid.to(device=query.device, dtype=torch.bool).reshape(batch * heads, selected).contiguous(),
        rank_maps[0].to(device=query.device, dtype=torch.int32).contiguous(),
        rank_maps[1].to(device=query.device, dtype=torch.int32).contiguous(),
        tail_key.reshape(batch * heads, tail_key.shape[2], channels).contiguous(),
        tail_value.reshape(batch * heads, tail_value.shape[2], channels).contiguous(),
        source_selected_output.float().reshape(batch * heads, channels).contiguous(),
        global_mass.float().reshape(batch * heads).contiguous(),
        int(current_position),
        float(scaling),
        64,
        heads,
        heads,
        int(loop),
        tuple(int(stream.original_shape[2]) for stream in keys),
        channels,
        bool(return_current_logits),
    )
    output = raw_outputs[0].reshape(batch, heads, 1, channels).float()
    if return_current_logits:
        return output, raw_outputs[1].reshape(batch, heads).float()
    return output


def kivi_fused_dense_source_attention(
    *,
    query: torch.Tensor,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    key_streams: tuple[KiviPacked4Bit | None, ...],
    value_streams: tuple[KiviPacked4Bit | None, ...],
    rank_maps: tuple[torch.Tensor, torch.Tensor],
    tail_key: torch.Tensor,
    tail_value: torch.Tensor,
    scaling: float,
    loop: int,
    extension: Any,
    require_cuda: bool = True,
    return_logits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fuse full Loop-1/2 KIVI QK, softmax, and PV in one launch."""

    if int(loop) not in (0, 1):
        raise ValueError("dense source attention is only valid for Loop 1/2")
    keys = _prepare_fused_cross_loop_streams(key_streams, axis="per_channel")
    values = _prepare_fused_cross_loop_streams(value_streams, axis="per_token")
    anchor = keys[0]
    batch, heads, base_tokens, channels = anchor.original_shape
    expected_vector = (batch, heads, 1, channels)
    if any(tuple(tensor.shape) != expected_vector for tensor in (query, current_key, current_value)):
        raise ValueError("dense attention vectors differ from the KIVI anchor")
    if any(stream.original_shape[:2] != (batch, heads) for stream in (*keys, *values)):
        raise ValueError("dense attention streams differ from the KIVI anchor")
    if tuple(tail_key.shape[:2]) != (batch, heads) or tuple(tail_value.shape) != tuple(tail_key.shape):
        raise ValueError("tail K/V shape differs from the KIVI anchor")
    if int(tail_key.shape[-1]) != channels:
        raise ValueError("tail K/V channels differ from the KIVI anchor")
    if any(tensor.dtype != torch.bfloat16 for tensor in (query, current_key, current_value, tail_key, tail_value)):
        raise ValueError("fused dense attention currently specializes BF16 vectors")
    if require_cuda and (not query.is_cuda or not current_key.is_cuda or not anchor.payload.is_cuda):
        raise RuntimeError("fused KIVI dense source attention requires CUDA tensors")
    if any(int(stream.original_shape[2]) != base_tokens for stream in keys[:2]):
        raise ValueError("Loop-1/2 K streams must share the anchor token count")

    prepared_query = query.to(torch.float16).reshape(batch * heads, 1, channels).contiguous()
    key_payloads = [
        stream.payload.reshape(batch * heads, stream.payload.shape[2], channels).contiguous()
        for stream in keys
    ]
    key_scales = [
        stream.scale.reshape(batch * heads, stream.scale.shape[2], channels).contiguous()
        for stream in keys
    ]
    key_minima = [
        stream.minimum.reshape(batch * heads, stream.minimum.shape[2], channels).contiguous()
        for stream in keys
    ]
    value_payloads = [
        stream.payload.reshape(batch * heads, stream.payload.shape[2], stream.original_shape[2]).contiguous()
        for stream in values
    ]
    value_scales = [
        stream.scale.reshape(batch * heads, stream.scale.shape[2], stream.original_shape[2]).contiguous()
        for stream in values
    ]
    value_minima = [
        stream.minimum.reshape(batch * heads, stream.minimum.shape[2], stream.original_shape[2]).contiguous()
        for stream in values
    ]
    raw_outputs = extension.dense_source_attention_cuda(
        prepared_query,
        query.reshape(batch * heads, channels).contiguous(),
        current_key.reshape(batch * heads, channels).contiguous(),
        current_value.reshape(batch * heads, channels).contiguous(),
        key_payloads,
        key_scales,
        key_minima,
        value_payloads,
        value_scales,
        value_minima,
        rank_maps[0].to(device=query.device, dtype=torch.int32).contiguous(),
        rank_maps[1].to(device=query.device, dtype=torch.int32).contiguous(),
        tail_key.reshape(batch * heads, tail_key.shape[2], channels).contiguous(),
        tail_value.reshape(batch * heads, tail_value.shape[2], channels).contiguous(),
        float(scaling),
        64,
        heads,
        heads,
        int(loop),
        tuple(int(stream.original_shape[2]) for stream in keys),
        channels,
        bool(return_logits),
    )
    output = raw_outputs[0].reshape(batch, heads, 1, channels).float()
    logits = None
    if return_logits:
        logits = raw_outputs[1].reshape(batch, heads, 1, base_tokens + int(tail_key.shape[2]) + 1).float()
    return output, logits


def kivi_cross_loop_qk(
    query: torch.Tensor,
    streams: tuple[KiviPacked4Bit | None, ...],
    rank_maps: tuple[torch.Tensor, torch.Tensor],
    indices: torch.Tensor,
    *,
    loop: int,
    gemv: Callable[[torch.Tensor, KiviPacked4Bit], torch.Tensor],
    selected_gemv: Callable[
        [torch.Tensor, KiviPacked4Bit, torch.Tensor], torch.Tensor
    ]
    | None = None,
) -> torch.Tensor:
    """Read selected logits from an anchor plus nested K delta streams."""

    anchor, logical = _validate_cross_loop_reader(
        streams,
        rank_maps,
        indices,
        loop=loop,
        axis="per_channel",
    )
    if tuple(query.shape) != (*anchor.original_shape[:2], 1, anchor.original_shape[3]):
        raise ValueError("query shape differs from the K anchor")
    output = torch.zeros(
        (*anchor.original_shape[:2], 1, int(logical.shape[-1])),
        device=query.device,
        dtype=torch.float32,
    )
    for stream_index in range(int(loop) + 1):
        stream = streams[stream_index]
        if stream is None:
            continue
        if stream_index < 2:
            active = logical < anchor.original_shape[2]
            safe = torch.where(active, logical, torch.zeros_like(logical))
        else:
            safe_logical = torch.where(
                logical < anchor.original_shape[2],
                logical,
                torch.zeros_like(logical),
            )
            ranks = rank_maps[stream_index - 2].to(logical.device)[safe_logical]
            active = (logical < anchor.original_shape[2]) & (ranks >= 0)
            safe = torch.where(active, ranks, torch.zeros_like(ranks)).long()
        if selected_gemv is None:
            logits = gemv(query, stream).float()
            contribution = torch.gather(logits, -1, safe[:, :, None, :])
        else:
            contribution = selected_gemv(query, stream, safe).float()
        output += contribution * active[:, :, None, :]
    return output


def kivi_cross_loop_pv(
    weights: torch.Tensor,
    streams: tuple[KiviPacked4Bit | None, ...],
    rank_maps: tuple[torch.Tensor, torch.Tensor],
    indices: torch.Tensor,
    *,
    loop: int,
    gemv: Callable[[torch.Tensor, KiviPacked4Bit], torch.Tensor],
    workspaces: tuple[torch.Tensor | None, ...] | None = None,
    selected_gemv: Callable[
        [torch.Tensor, KiviPacked4Bit, torch.Tensor], torch.Tensor
    ]
    | None = None,
) -> torch.Tensor:
    """Read a weighted V sum from an anchor plus nested compact deltas."""

    anchor, logical = _validate_cross_loop_reader(
        streams,
        rank_maps,
        indices,
        loop=loop,
        axis="per_token",
    )
    expected_weights = (*anchor.original_shape[:2], 1, int(logical.shape[-1]))
    if tuple(weights.shape) != expected_weights:
        raise ValueError("weights shape differs from selected logical rows")
    if workspaces is None:
        workspaces = (None, None, None, None)
    if len(workspaces) != 4:
        raise ValueError("four optional PV input workspaces are required")
    output = torch.zeros(
        (*anchor.original_shape[:2], 1, anchor.original_shape[3]),
        device=weights.device,
        dtype=torch.float32,
    )
    for stream_index in range(int(loop) + 1):
        stream = streams[stream_index]
        if stream is None:
            continue
        input_rows = stream.original_shape[2]
        workspace = workspaces[stream_index]
        required_shape = (*anchor.original_shape[:2], 1, input_rows)
        if workspace is None:
            prepared = torch.zeros(
                required_shape,
                device=weights.device,
                dtype=torch.float32,
            )
        else:
            if tuple(workspace.shape) != required_shape or workspace.device != weights.device:
                raise ValueError("PV input workspace shape/device differs from its stream")
            prepared = workspace.zero_()
        if stream_index < 2:
            active = logical < anchor.original_shape[2]
            safe = torch.where(active, logical, torch.zeros_like(logical))
        else:
            safe_logical = torch.where(
                logical < anchor.original_shape[2],
                logical,
                torch.zeros_like(logical),
            )
            ranks = rank_maps[stream_index - 2].to(logical.device)[safe_logical]
            active = (logical < anchor.original_shape[2]) & (ranks >= 0)
            safe = torch.where(active, ranks, torch.zeros_like(ranks)).long()
        masked_weights = weights.float() * active[:, :, None, :]
        if selected_gemv is None:
            prepared.scatter_add_(-1, safe[:, :, None, :], masked_weights)
            output += gemv(prepared, stream).float()
        else:
            output += selected_gemv(masked_weights, stream, safe).float()
    return output
