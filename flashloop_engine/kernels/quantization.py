"""Physical 4-bit packing and selected-row dequantization kernels.

The layout follows FlashLoop's Torch reference: two adjacent channels share one
byte, the even channel occupies the low nibble, and affine metadata is FP16.
The CUDA reader allocates only the requested rows; it never materializes a full
dequantized cache.
"""

from __future__ import annotations

import torch

from ..packing import Packed4Bit, pack_k_per_channel, pack_v_per_token

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by CPU-only installations
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


_QK_READER_DO_NOT_SPECIALIZE = (
    "selected_rows",
    "prefix_tokens",
    "tail_tokens",
    "groups0",
    "groups1",
    "groups2",
    "groups3",
    "tokens2",
    "tokens3",
)
_PV_READER_DO_NOT_SPECIALIZE = (
    "selected_rows",
    "prefix_tokens",
    "tail_tokens",
    "tokens2",
    "tokens3",
    "channel_groups",
)
_CROSS_LOOP_DO_NOT_SPECIALIZE = (
    "selected_rows",
    "prefix_tokens",
    "tail_tokens",
    "current_position",
    "key_groups0",
    "key_groups1",
    "key_groups2",
    "key_groups3",
    "tokens2",
    "tokens3",
    "value_channel_groups",
)
_DENSE_SOURCE_DO_NOT_SPECIALIZE = (
    "prefix_tokens",
    "tail_tokens",
    "tail_stride",
    "key_groups0",
    "key_groups1",
    "value_channel_groups",
)


def _split_kv_geometry(total_rows: int, *, split_rows: int) -> tuple[int, int]:
    """Return the number of token splits and the merge reduction width.

    The merge width is rounded to a power of two because Triton reductions
    require a static block.  Token splits give q=1 decode enough independent
    programs to occupy the GPU, unlike a one-program-per-head fused reader.
    """

    total_rows = int(total_rows)
    split_rows = int(split_rows)
    if total_rows <= 0 or split_rows <= 0:
        raise ValueError("split-KV dimensions must be positive")
    splits = (total_rows + split_rows - 1) // split_rows
    merge_width = 1 << (splits - 1).bit_length()
    return splits, merge_width


def _cross_loop_pv_grid(
    batch_heads: int,
    channels: int,
    *,
    block_channels: int,
) -> tuple[int, int]:
    """Tile PV independently over channels, following KIVI's GEMV schedule."""

    if batch_heads <= 0 or channels <= 0 or block_channels <= 0:
        raise ValueError("PV launch dimensions must be positive")
    return int(batch_heads), (int(channels) + int(block_channels) - 1) // int(
        block_channels
    )


def _validate_cross_loop_pv_tiling(
    channels: int,
    block_channels: int,
    num_warps: int,
) -> tuple[int, int]:
    """Validate a KIVI-style output-channel GEMV tile."""

    channels = int(channels)
    block_channels = int(block_channels)
    num_warps = int(num_warps)
    if channels <= 0 or block_channels <= 0 or channels % block_channels:
        raise ValueError("PV block channels must positively divide head channels")
    if num_warps not in (1, 2, 4, 8):
        raise ValueError("PV num warps must be one of 1, 2, 4, or 8")
    return block_channels, num_warps


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
    def _k_minmax_kernel(
        values,
        work_minimum,
        work_scale,
        stored_minimum,
        stored_scale,
        tokens,
        channels: tl.constexpr,
        token_groups,
        group_size: tl.constexpr,
        metadata_count,
        BLOCK_ROWS: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        valid_rows = rows < metadata_count
        channel = rows % channels
        partial = rows // channels
        group = partial % token_groups
        head_batch = partial // token_groups
        group_offsets = tl.arange(0, group_size)
        token = group[:, None] * group_size + group_offsets[None, :]
        offsets = (
            head_batch[:, None] * tokens * channels
            + token * channels
            + channel[:, None]
        )
        valid = valid_rows[:, None] & (token < tokens)
        values_block = tl.load(values + offsets, mask=valid, other=0.0).to(tl.float32)
        minimum = tl.min(tl.where(valid, values_block, float("inf")), axis=1)
        maximum = tl.max(tl.where(valid, values_block, -float("inf")), axis=1)
        span = maximum - minimum
        scale = tl.where(span > 0.0, span / 15.0, 1.0)
        tl.store(work_minimum + rows, minimum, mask=valid_rows)
        tl.store(work_scale + rows, scale, mask=valid_rows)
        tl.store(stored_minimum + rows, minimum, mask=valid_rows)
        tl.store(stored_scale + rows, scale, mask=valid_rows)


    @triton.jit
    def _v_minmax_kernel(
        values,
        work_minimum,
        work_scale,
        stored_minimum,
        stored_scale,
        tokens,
        channels: tl.constexpr,
        channel_groups,
        group_size: tl.constexpr,
        metadata_count,
        BLOCK_ROWS: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        valid_rows = rows < metadata_count
        channel_group = rows % channel_groups
        partial = rows // channel_groups
        token = partial % tokens
        head_batch = partial // tokens
        group_offsets = tl.arange(0, group_size)
        channel = channel_group[:, None] * group_size + group_offsets[None, :]
        offsets = (
            head_batch[:, None] * tokens * channels
            + token[:, None] * channels
            + channel
        )
        valid = valid_rows[:, None] & (channel < channels)
        values_block = tl.load(values + offsets, mask=valid, other=0.0).to(tl.float32)
        minimum = tl.min(tl.where(valid, values_block, float("inf")), axis=1)
        maximum = tl.max(tl.where(valid, values_block, -float("inf")), axis=1)
        span = maximum - minimum
        scale = tl.where(span > 0.0, span / 15.0, 1.0)
        tl.store(work_minimum + rows, minimum, mask=valid_rows)
        tl.store(work_scale + rows, scale, mask=valid_rows)
        tl.store(stored_minimum + rows, minimum, mask=valid_rows)
        tl.store(stored_scale + rows, scale, mask=valid_rows)


    @triton.jit
    def _pack_nibbles_kernel(
        values,
        work_minimum,
        work_scale,
        payload,
        total_bytes,
        tokens: tl.constexpr,
        channels: tl.constexpr,
        token_groups: tl.constexpr,
        channel_groups: tl.constexpr,
        group_size: tl.constexpr,
        PER_CHANNEL: tl.constexpr,
        BLOCK_BYTES: tl.constexpr,
    ):
        byte_offsets = tl.program_id(0) * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
        valid = byte_offsets < total_bytes
        channel_pairs = (channels + 1) // 2
        pair = byte_offsets % channel_pairs
        partial = byte_offsets // channel_pairs
        token = partial % tokens
        head_batch = partial // tokens
        channel0 = pair * 2
        channel1 = channel0 + 1
        value_base = head_batch * tokens * channels + token * channels

        if PER_CHANNEL:
            meta_base = (head_batch * token_groups + token // group_size) * channels
            meta0 = meta_base + channel0
            meta1 = meta_base + channel1
        else:
            meta_base = (head_batch * tokens + token) * channel_groups
            meta0 = meta_base + channel0 // group_size
            meta1 = meta_base + channel1 // group_size

        minimum0 = tl.load(work_minimum + meta0, mask=valid, other=0.0)
        scale0 = tl.load(work_scale + meta0, mask=valid, other=1.0)
        value0 = tl.load(values + value_base + channel0, mask=valid, other=0.0).to(tl.float32)
        q0 = tl.maximum(
            0.0,
            tl.minimum(15.0, _subtract_divide_round_nearest(value0, minimum0, scale0)),
        )
        code0 = _round_nearest_even_int32(q0)

        valid1 = valid & (channel1 < channels)
        minimum1 = tl.load(work_minimum + meta1, mask=valid1, other=0.0)
        scale1 = tl.load(work_scale + meta1, mask=valid1, other=1.0)
        value1 = tl.load(values + value_base + channel1, mask=valid1, other=0.0).to(tl.float32)
        q1 = tl.maximum(
            0.0,
            tl.minimum(15.0, _subtract_divide_round_nearest(value1, minimum1, scale1)),
        )
        code1 = _round_nearest_even_int32(q1)
        packed = code0 | (code1 << 4)
        tl.store(payload + byte_offsets, packed, mask=valid)


    @triton.jit
    def _gather_dequantize_kernel(
        payload,
        minimum,
        scale,
        row_groups,
        indices,
        output,
        selected_rows: tl.constexpr,
        tokens: tl.constexpr,
        channels: tl.constexpr,
        token_groups: tl.constexpr,
        channel_groups: tl.constexpr,
        group_size: tl.constexpr,
        SHARED_INDICES: tl.constexpr,
        PER_CHANNEL: tl.constexpr,
        HAS_ROW_GROUPS: tl.constexpr,
        BLOCK_CHANNELS: tl.constexpr,
    ):
        selected_linear = tl.program_id(0)
        channel_block = tl.program_id(1)
        selected = selected_linear % selected_rows
        head_batch = selected_linear // selected_rows
        if SHARED_INDICES:
            token = tl.load(indices + selected).to(tl.int32)
        else:
            token = tl.load(indices + selected_linear).to(tl.int32)
        channel = channel_block * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
        valid_channel = channel < channels
        byte_offset = (
            (head_batch * tokens + token) * ((channels + 1) // 2)
            + channel // 2
        )
        packed = tl.load(payload + byte_offset, mask=valid_channel, other=0).to(tl.int32)
        shift = (channel & 1) * 4
        code = (packed >> shift) & 15

        if PER_CHANNEL:
            if HAS_ROW_GROUPS:
                group = tl.load(row_groups + token).to(tl.int32)
            else:
                group = token // group_size
            metadata_offset = (head_batch * token_groups + group) * channels + channel
        else:
            group = channel // group_size
            metadata_offset = (head_batch * tokens + token) * channel_groups + group
        row_minimum = tl.load(minimum + metadata_offset, mask=valid_channel, other=0.0)
        row_scale = tl.load(scale + metadata_offset, mask=valid_channel, other=1.0)
        decoded = (code.to(tl.float32) * row_scale + row_minimum).to(tl.float16).to(
            tl.float32
        )
        output_offset = selected_linear * channels + channel
        tl.store(output + output_offset, decoded, mask=valid_channel)


    @triton.jit
    def _packed_qk_kernel(
        query,
        payload,
        minimum,
        scale,
        row_groups,
        indices,
        output,
        selected_rows: tl.constexpr,
        tokens: tl.constexpr,
        channels: tl.constexpr,
        token_groups: tl.constexpr,
        group_size: tl.constexpr,
        SHARED_INDICES: tl.constexpr,
        HAS_ROW_GROUPS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        head_batch = tl.program_id(0)
        row_block = tl.program_id(1)
        selected = row_block * BLOCK_N + tl.arange(0, BLOCK_N)
        valid_row = selected < selected_rows
        if SHARED_INDICES:
            token = tl.load(indices + selected, mask=valid_row, other=0).to(tl.int32)
        else:
            token = tl.load(
                indices + head_batch * selected_rows + selected,
                mask=valid_row,
                other=0,
            ).to(tl.int32)
        channel = tl.arange(0, BLOCK_D)
        valid_channel = channel < channels
        byte_offset = (
            (head_batch * tokens + token[:, None]) * ((channels + 1) // 2)
            + channel[None, :] // 2
        )
        packed = tl.load(
            payload + byte_offset,
            mask=valid_row[:, None] & valid_channel[None, :],
            other=0,
        ).to(tl.int32)
        code = (packed >> ((channel[None, :] & 1) * 4)) & 15
        if HAS_ROW_GROUPS:
            group = tl.load(row_groups + token, mask=valid_row, other=0).to(tl.int32)
        else:
            group = token // group_size
        metadata_offset = (
            (head_batch * token_groups + group[:, None]) * channels
            + channel[None, :]
        )
        row_minimum = tl.load(
            minimum + metadata_offset,
            mask=valid_row[:, None] & valid_channel[None, :],
            other=0.0,
        ).to(tl.float32)
        row_scale = tl.load(
            scale + metadata_offset,
            mask=valid_row[:, None] & valid_channel[None, :],
            other=1.0,
        ).to(tl.float32)
        decoded = (code.to(tl.float32) * row_scale + row_minimum).to(tl.float16).to(
            tl.float32
        )
        query_row = tl.load(
            query + head_batch * channels + channel,
            mask=valid_channel,
            other=0.0,
        ).to(tl.float32)
        products = tl.where(
            valid_row[:, None] & valid_channel[None, :],
            decoded * query_row[None, :],
            0.0,
        )
        result = tl.sum(products, axis=1)
        tl.store(
            output + head_batch * selected_rows + selected,
            result,
            mask=valid_row,
        )


    @triton.jit
    def _packed_pv_kernel(
        weights,
        payload,
        minimum,
        scale,
        indices,
        output,
        selected_rows: tl.constexpr,
        tokens: tl.constexpr,
        channels: tl.constexpr,
        channel_groups: tl.constexpr,
        group_size: tl.constexpr,
        SHARED_INDICES: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        head_batch = tl.program_id(0)
        channel = tl.arange(0, BLOCK_D)
        valid_channel = channel < channels
        accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for start in range(0, selected_rows, BLOCK_N):
            selected = start + tl.arange(0, BLOCK_N)
            valid_row = selected < selected_rows
            if SHARED_INDICES:
                token = tl.load(indices + selected, mask=valid_row, other=0).to(tl.int32)
            else:
                token = tl.load(
                    indices + head_batch * selected_rows + selected,
                    mask=valid_row,
                    other=0,
                ).to(tl.int32)
            byte_offset = (
                (head_batch * tokens + token[:, None]) * ((channels + 1) // 2)
                + channel[None, :] // 2
            )
            packed = tl.load(
                payload + byte_offset,
                mask=valid_row[:, None] & valid_channel[None, :],
                other=0,
            ).to(tl.int32)
            code = (packed >> ((channel[None, :] & 1) * 4)) & 15
            channel_group = channel // group_size
            metadata_offset = (
                (head_batch * tokens + token[:, None]) * channel_groups
                + channel_group[None, :]
            )
            row_minimum = tl.load(
                minimum + metadata_offset,
                mask=valid_row[:, None] & valid_channel[None, :],
                other=0.0,
            ).to(tl.float32)
            row_scale = tl.load(
                scale + metadata_offset,
                mask=valid_row[:, None] & valid_channel[None, :],
                other=1.0,
            ).to(tl.float32)
            decoded = (code.to(tl.float32) * row_scale + row_minimum).to(tl.float16).to(
                tl.float32
            )
            row_weight = tl.load(
                weights + head_batch * selected_rows + selected,
                mask=valid_row,
                other=0.0,
            ).to(tl.float32)
            accumulator += tl.sum(
                tl.where(
                    valid_row[:, None] & valid_channel[None, :],
                    row_weight[:, None] * decoded,
                    0.0,
                ),
                axis=0,
            )
        tl.store(
            output + head_batch * channels + channel,
            accumulator,
            mask=valid_channel,
        )


    @triton.jit
    def _load_cross_loop_k(
        payload,
        minimum,
        scale,
        row_groups,
        token,
        channel,
        valid,
        tokens,
        channels: tl.constexpr,
        token_groups,
        group_size: tl.constexpr,
        HAS_ROW_GROUPS: tl.constexpr,
    ):
        byte_offset = token[:, None] * ((channels + 1) // 2) + channel[None, :] // 2
        packed = tl.load(payload + byte_offset, mask=valid, other=0).to(tl.int32)
        code = (packed >> ((channel[None, :] & 1) * 4)) & 15
        if HAS_ROW_GROUPS:
            row_valid = tl.max(valid, axis=1) != 0
            group = tl.load(row_groups + token, mask=row_valid, other=0).to(tl.int32)
        else:
            group = token // group_size
        metadata_offset = group[:, None] * channels + channel[None, :]
        row_minimum = tl.load(minimum + metadata_offset, mask=valid, other=0.0).to(
            tl.float32
        )
        row_scale = tl.load(scale + metadata_offset, mask=valid, other=1.0).to(
            tl.float32
        )
        return (code.to(tl.float32) * row_scale + row_minimum).to(tl.float16).to(
            tl.float32
        )


    @triton.jit
    def _load_cross_loop_v(
        payload,
        minimum,
        scale,
        token,
        channel,
        valid,
        tokens,
        channels: tl.constexpr,
        channel_groups,
        group_size: tl.constexpr,
    ):
        byte_offset = token[:, None] * ((channels + 1) // 2) + channel[None, :] // 2
        packed = tl.load(payload + byte_offset, mask=valid, other=0).to(tl.int32)
        code = (packed >> ((channel[None, :] & 1) * 4)) & 15
        metadata_offset = (
            token[:, None] * channel_groups + channel[None, :] // group_size
        )
        row_minimum = tl.load(minimum + metadata_offset, mask=valid, other=0.0).to(
            tl.float32
        )
        row_scale = tl.load(scale + metadata_offset, mask=valid, other=1.0).to(
            tl.float32
        )
        return (code.to(tl.float32) * row_scale + row_minimum).to(tl.float16).to(
            tl.float32
        )


    @triton.jit(do_not_specialize=_QK_READER_DO_NOT_SPECIALIZE)
    def _cross_loop_qk_kernel(
        query,
        indices,
        payload0,
        minimum0,
        scale0,
        row_groups0,
        payload1,
        minimum1,
        scale1,
        row_groups1,
        payload2,
        minimum2,
        scale2,
        row_groups2,
        payload3,
        minimum3,
        scale3,
        row_groups3,
        absolute2,
        absolute3,
        rank3,
        rank4,
        tail,
        output,
        selected_rows,
        prefix_tokens,
        tail_tokens,
        channels: tl.constexpr,
        groups0,
        groups1,
        groups2,
        groups3,
        tokens2,
        tokens3,
        group_size: tl.constexpr,
        LOOP: tl.constexpr,
        HAS2: tl.constexpr,
        HAS3: tl.constexpr,
        ABSOLUTE_OVERRIDES: tl.constexpr,
        BF16_ABSOLUTE_OVERRIDES: tl.constexpr,
        ROW_GROUPS0: tl.constexpr,
        ROW_GROUPS1: tl.constexpr,
        ROW_GROUPS2: tl.constexpr,
        ROW_GROUPS3: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        head_batch = tl.program_id(0)
        selected = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        valid_row = selected < selected_rows
        logical = tl.load(
            indices + head_batch * selected_rows + selected,
            mask=valid_row,
            other=0,
        ).to(tl.int32)
        is_prefix = valid_row & (logical < prefix_tokens)
        prefix_token = tl.where(is_prefix, logical, 0)
        channel = tl.arange(0, BLOCK_D)
        valid_channel = channel < channels
        valid_prefix = is_prefix[:, None] & valid_channel[None, :]
        head_prefix = head_batch * prefix_tokens
        decoded = _load_cross_loop_k(
            payload0 + head_prefix * ((channels + 1) // 2),
            minimum0 + head_batch * groups0 * channels,
            scale0 + head_batch * groups0 * channels,
            row_groups0,
            prefix_token,
            channel,
            valid_prefix,
            prefix_tokens,
            channels,
            groups0,
            group_size,
            ROW_GROUPS0,
        )
        if LOOP >= 1:
            decoded += _load_cross_loop_k(
                payload1 + head_prefix * ((channels + 1) // 2),
                minimum1 + head_batch * groups1 * channels,
                scale1 + head_batch * groups1 * channels,
                row_groups1,
                prefix_token,
                channel,
                valid_prefix,
                prefix_tokens,
                channels,
                groups1,
                group_size,
                ROW_GROUPS1,
            )
        if LOOP >= 2 and HAS2:
            compact2 = tl.load(rank3 + prefix_token, mask=is_prefix, other=-1).to(tl.int32)
            active2 = is_prefix & (compact2 >= 0)
            safe2 = tl.where(active2, compact2, 0)
            if BF16_ABSOLUTE_OVERRIDES:
                update2 = tl.load(
                    absolute2
                    + (head_batch * tokens2 + safe2[:, None]) * channels
                    + channel[None, :],
                    mask=active2[:, None] & valid_channel[None, :],
                    other=0.0,
                ).to(tl.float32)
            else:
                update2 = _load_cross_loop_k(
                    payload2 + head_batch * tokens2 * ((channels + 1) // 2),
                    minimum2 + head_batch * groups2 * channels,
                    scale2 + head_batch * groups2 * channels,
                    row_groups2,
                    safe2,
                    channel,
                    active2[:, None] & valid_channel[None, :],
                    1,
                    channels,
                    groups2,
                    group_size,
                    ROW_GROUPS2,
                )
            if ABSOLUTE_OVERRIDES:
                decoded = tl.where(active2[:, None], update2, decoded)
            else:
                decoded += update2
        if LOOP >= 3 and HAS3:
            compact3 = tl.load(rank4 + prefix_token, mask=is_prefix, other=-1).to(tl.int32)
            active3 = is_prefix & (compact3 >= 0)
            safe3 = tl.where(active3, compact3, 0)
            if BF16_ABSOLUTE_OVERRIDES:
                update3 = tl.load(
                    absolute3
                    + (head_batch * tokens3 + safe3[:, None]) * channels
                    + channel[None, :],
                    mask=active3[:, None] & valid_channel[None, :],
                    other=0.0,
                ).to(tl.float32)
            else:
                update3 = _load_cross_loop_k(
                    payload3 + head_batch * tokens3 * ((channels + 1) // 2),
                    minimum3 + head_batch * groups3 * channels,
                    scale3 + head_batch * groups3 * channels,
                    row_groups3,
                    safe3,
                    channel,
                    active3[:, None] & valid_channel[None, :],
                    1,
                    channels,
                    groups3,
                    group_size,
                    ROW_GROUPS3,
                )
            if ABSOLUTE_OVERRIDES:
                decoded = tl.where(active3[:, None], update3, decoded)
            else:
                decoded += update3
        is_tail = valid_row & ~is_prefix
        tail_token = tl.where(is_tail, logical - prefix_tokens, 0)
        tail_offset = (
            (head_batch * tail_tokens + tail_token[:, None]) * channels
            + channel[None, :]
        )
        decoded = tl.where(
            is_tail[:, None] & valid_channel[None, :],
            tl.load(
                tail + tail_offset,
                mask=is_tail[:, None] & valid_channel[None, :],
                other=0.0,
            ).to(tl.float32),
            decoded,
        )
        query_row = tl.load(
            query + head_batch * channels + channel,
            mask=valid_channel,
            other=0.0,
        ).to(tl.float32)
        result = tl.sum(decoded * query_row[None, :], axis=1)
        tl.store(
            output + head_batch * selected_rows + selected,
            result,
            mask=valid_row,
        )


    @triton.jit(do_not_specialize=_PV_READER_DO_NOT_SPECIALIZE)
    def _cross_loop_pv_kernel(
        weights,
        indices,
        payload0,
        minimum0,
        scale0,
        payload1,
        minimum1,
        scale1,
        payload2,
        minimum2,
        scale2,
        payload3,
        minimum3,
        scale3,
        absolute2,
        absolute3,
        rank3,
        rank4,
        tail,
        output,
        selected_rows,
        prefix_tokens,
        tail_tokens,
        channels: tl.constexpr,
        tokens2,
        tokens3,
        channel_groups,
        group_size: tl.constexpr,
        LOOP: tl.constexpr,
        HAS2: tl.constexpr,
        HAS3: tl.constexpr,
        ABSOLUTE_OVERRIDES: tl.constexpr,
        BF16_ABSOLUTE_OVERRIDES: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        head_batch = tl.program_id(0)
        channel = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
        valid_channel = channel < channels
        accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
        start = 0
        while start < selected_rows:
            selected = start + tl.arange(0, BLOCK_N)
            valid_row = selected < selected_rows
            logical = tl.load(
                indices + head_batch * selected_rows + selected,
                mask=valid_row,
                other=0,
            ).to(tl.int32)
            row_weight = tl.load(
                weights + head_batch * selected_rows + selected,
                mask=valid_row,
                other=0.0,
            ).to(tl.float32)
            is_prefix = valid_row & (logical < prefix_tokens)
            prefix_token = tl.where(is_prefix, logical, 0)
            valid_prefix = is_prefix[:, None] & valid_channel[None, :]
            head_prefix = head_batch * prefix_tokens
            decoded = _load_cross_loop_v(
                payload0 + head_prefix * ((channels + 1) // 2),
                minimum0 + head_batch * prefix_tokens * channel_groups,
                scale0 + head_batch * prefix_tokens * channel_groups,
                prefix_token,
                channel,
                valid_prefix,
                prefix_tokens,
                channels,
                channel_groups,
                group_size,
            )
            if LOOP >= 1:
                decoded += _load_cross_loop_v(
                    payload1 + head_prefix * ((channels + 1) // 2),
                    minimum1 + head_batch * prefix_tokens * channel_groups,
                    scale1 + head_batch * prefix_tokens * channel_groups,
                    prefix_token,
                    channel,
                    valid_prefix,
                    prefix_tokens,
                    channels,
                    channel_groups,
                    group_size,
                )
            if LOOP >= 2 and HAS2:
                compact2 = tl.load(rank3 + prefix_token, mask=is_prefix, other=-1).to(
                    tl.int32
                )
                active2 = is_prefix & (compact2 >= 0)
                safe2 = tl.where(active2, compact2, 0)
                if BF16_ABSOLUTE_OVERRIDES:
                    update2 = tl.load(
                        absolute2
                        + (head_batch * tokens2 + safe2[:, None]) * channels
                        + channel[None, :],
                        mask=active2[:, None] & valid_channel[None, :],
                        other=0.0,
                    ).to(tl.float32)
                else:
                    update2 = _load_cross_loop_v(
                        payload2 + head_batch * tokens2 * ((channels + 1) // 2),
                        minimum2 + head_batch * tokens2 * channel_groups,
                        scale2 + head_batch * tokens2 * channel_groups,
                        safe2,
                        channel,
                        active2[:, None] & valid_channel[None, :],
                        tokens2,
                        channels,
                        channel_groups,
                        group_size,
                    )
                if ABSOLUTE_OVERRIDES:
                    decoded = tl.where(active2[:, None], update2, decoded)
                else:
                    decoded += update2
            if LOOP >= 3 and HAS3:
                compact3 = tl.load(rank4 + prefix_token, mask=is_prefix, other=-1).to(
                    tl.int32
                )
                active3 = is_prefix & (compact3 >= 0)
                safe3 = tl.where(active3, compact3, 0)
                if BF16_ABSOLUTE_OVERRIDES:
                    update3 = tl.load(
                        absolute3
                        + (head_batch * tokens3 + safe3[:, None]) * channels
                        + channel[None, :],
                        mask=active3[:, None] & valid_channel[None, :],
                        other=0.0,
                    ).to(tl.float32)
                else:
                    update3 = _load_cross_loop_v(
                        payload3 + head_batch * tokens3 * ((channels + 1) // 2),
                        minimum3 + head_batch * tokens3 * channel_groups,
                        scale3 + head_batch * tokens3 * channel_groups,
                        safe3,
                        channel,
                        active3[:, None] & valid_channel[None, :],
                        tokens3,
                        channels,
                        channel_groups,
                        group_size,
                    )
                if ABSOLUTE_OVERRIDES:
                    decoded = tl.where(active3[:, None], update3, decoded)
                else:
                    decoded += update3
            is_tail = valid_row & ~is_prefix
            tail_token = tl.where(is_tail, logical - prefix_tokens, 0)
            tail_offset = (
                (head_batch * tail_tokens + tail_token[:, None]) * channels
                + channel[None, :]
            )
            decoded = tl.where(
                is_tail[:, None] & valid_channel[None, :],
                tl.load(
                    tail + tail_offset,
                    mask=is_tail[:, None] & valid_channel[None, :],
                    other=0.0,
                ).to(tl.float32),
                decoded,
            )
            accumulator += tl.sum(
                tl.where(
                    valid_row[:, None] & valid_channel[None, :],
                    row_weight[:, None] * decoded,
                    0.0,
                ),
                axis=0,
            )
            start += BLOCK_N
        tl.store(
            output + head_batch * channels + channel,
            accumulator,
            mask=valid_channel,
        )


    @triton.jit(do_not_specialize=_CROSS_LOOP_DO_NOT_SPECIALIZE)
    def _cross_loop_sparse_attention_delta_kernel(
        query,
        current_key,
        current_value,
        indices,
        valid_indices,
        key_payload0,
        key_minimum0,
        key_scale0,
        key_row_groups0,
        key_payload1,
        key_minimum1,
        key_scale1,
        key_row_groups1,
        key_payload2,
        key_minimum2,
        key_scale2,
        key_row_groups2,
        key_payload3,
        key_minimum3,
        key_scale3,
        key_row_groups3,
        key_absolute2,
        key_absolute3,
        value_payload0,
        value_minimum0,
        value_scale0,
        value_payload1,
        value_minimum1,
        value_scale1,
        value_payload2,
        value_minimum2,
        value_scale2,
        value_payload3,
        value_minimum3,
        value_scale3,
        value_absolute2,
        value_absolute3,
        rank3,
        rank4,
        tail_key,
        tail_value,
        source_output,
        global_mass,
        output,
        partial_maximum,
        partial_denominator,
        partial_accumulator,
        selected_rows,
        prefix_tokens,
        tail_tokens,
        current_position,
        num_splits,
        scaling,
        channels: tl.constexpr,
        key_groups0,
        key_groups1,
        key_groups2,
        key_groups3,
        tokens2,
        tokens3,
        value_channel_groups,
        group_size: tl.constexpr,
        LOOP: tl.constexpr,
        HAS2: tl.constexpr,
        HAS3: tl.constexpr,
        ABSOLUTE_OVERRIDES: tl.constexpr,
        BF16_ABSOLUTE_OVERRIDES: tl.constexpr,
        ROW_GROUPS0: tl.constexpr,
        ROW_GROUPS1: tl.constexpr,
        ROW_GROUPS2: tl.constexpr,
        ROW_GROUPS3: tl.constexpr,
        SPLIT_KV: tl.constexpr,
        SPLIT_N: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Online-softmax sparse attention over packed cross-loop K/V.

        One program owns one batch/head pair.  This mirrors Chipmunk's fused
        sparse attention dataflow, while decoding KIVI-style packed anchor and
        delta streams directly instead of materializing an FP16 cache.
        """

        head_batch = tl.program_id(0)
        split = tl.program_id(1) if SPLIT_KV else 0
        channel = tl.arange(0, BLOCK_D)
        valid_channel = channel < channels
        query_row = tl.load(
            query + head_batch * channels + channel,
            mask=valid_channel,
            other=0.0,
        ).to(tl.float32)
        current_key_row = tl.load(
            current_key + head_batch * channels + channel,
            mask=valid_channel,
            other=0.0,
        ).to(tl.float32)
        current_value_row = tl.load(
            current_value + head_batch * channels + channel,
            mask=valid_channel,
            other=0.0,
        ).to(tl.float32)

        m_i = tl.full((1,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((1,), dtype=tl.float32)
        accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
        start = split * SPLIT_N if SPLIT_KV else 0
        stop = (
            tl.minimum(selected_rows, start + SPLIT_N)
            if SPLIT_KV
            else selected_rows
        )
        while start < stop:
            selected = start + tl.arange(0, BLOCK_N)
            in_range = selected < selected_rows
            row_valid = in_range & tl.load(
                valid_indices + head_batch * selected_rows + selected,
                mask=in_range,
                other=0,
            ).to(tl.int1)
            logical = tl.load(
                indices + head_batch * selected_rows + selected,
                mask=in_range,
                other=0,
            ).to(tl.int32)
            is_current = row_valid & (logical == current_position)
            is_prefix = row_valid & ~is_current & (logical < prefix_tokens)
            prefix_token = tl.where(is_prefix, logical, 0)
            valid_prefix = is_prefix[:, None] & valid_channel[None, :]
            head_prefix = head_batch * prefix_tokens

            decoded_key = _load_cross_loop_k(
                key_payload0 + head_prefix * ((channels + 1) // 2),
                key_minimum0 + head_batch * key_groups0 * channels,
                key_scale0 + head_batch * key_groups0 * channels,
                key_row_groups0,
                prefix_token,
                channel,
                valid_prefix,
                prefix_tokens,
                channels,
                key_groups0,
                group_size,
                ROW_GROUPS0,
            )
            if LOOP >= 1:
                decoded_key += _load_cross_loop_k(
                    key_payload1 + head_prefix * ((channels + 1) // 2),
                    key_minimum1 + head_batch * key_groups1 * channels,
                    key_scale1 + head_batch * key_groups1 * channels,
                    key_row_groups1,
                    prefix_token,
                    channel,
                    valid_prefix,
                    prefix_tokens,
                    channels,
                    key_groups1,
                    group_size,
                    ROW_GROUPS1,
                )
            if LOOP >= 2 and HAS2:
                compact2 = tl.load(rank3 + prefix_token, mask=is_prefix, other=-1).to(
                    tl.int32
                )
                active2 = is_prefix & (compact2 >= 0)
                safe2 = tl.where(active2, compact2, 0)
                if BF16_ABSOLUTE_OVERRIDES:
                    update_key2 = tl.load(
                        key_absolute2
                        + (head_batch * tokens2 + safe2[:, None]) * channels
                        + channel[None, :],
                        mask=active2[:, None] & valid_channel[None, :],
                        other=0.0,
                    ).to(tl.float32)
                else:
                    update_key2 = _load_cross_loop_k(
                        key_payload2 + head_batch * tokens2 * ((channels + 1) // 2),
                        key_minimum2 + head_batch * key_groups2 * channels,
                        key_scale2 + head_batch * key_groups2 * channels,
                        key_row_groups2,
                        safe2,
                        channel,
                        active2[:, None] & valid_channel[None, :],
                        1,
                        channels,
                        key_groups2,
                        group_size,
                        ROW_GROUPS2,
                    )
                if ABSOLUTE_OVERRIDES:
                    decoded_key = tl.where(
                        active2[:, None], update_key2, decoded_key
                    )
                else:
                    decoded_key += update_key2
            if LOOP >= 3 and HAS3:
                compact3 = tl.load(rank4 + prefix_token, mask=is_prefix, other=-1).to(
                    tl.int32
                )
                active3 = is_prefix & (compact3 >= 0)
                safe3 = tl.where(active3, compact3, 0)
                if BF16_ABSOLUTE_OVERRIDES:
                    update_key3 = tl.load(
                        key_absolute3
                        + (head_batch * tokens3 + safe3[:, None]) * channels
                        + channel[None, :],
                        mask=active3[:, None] & valid_channel[None, :],
                        other=0.0,
                    ).to(tl.float32)
                else:
                    update_key3 = _load_cross_loop_k(
                        key_payload3 + head_batch * tokens3 * ((channels + 1) // 2),
                        key_minimum3 + head_batch * key_groups3 * channels,
                        key_scale3 + head_batch * key_groups3 * channels,
                        key_row_groups3,
                        safe3,
                        channel,
                        active3[:, None] & valid_channel[None, :],
                        1,
                        channels,
                        key_groups3,
                        group_size,
                        ROW_GROUPS3,
                    )
                if ABSOLUTE_OVERRIDES:
                    decoded_key = tl.where(
                        active3[:, None], update_key3, decoded_key
                    )
                else:
                    decoded_key += update_key3
            is_tail = row_valid & ~is_current & ~is_prefix
            tail_token = tl.where(is_tail, logical - prefix_tokens, 0)
            tail_offset = (
                (head_batch * tail_tokens + tail_token[:, None]) * channels
                + channel[None, :]
            )
            decoded_key = tl.where(
                is_tail[:, None] & valid_channel[None, :],
                tl.load(
                    tail_key + tail_offset,
                    mask=is_tail[:, None] & valid_channel[None, :],
                    other=0.0,
                ).to(tl.float32),
                decoded_key,
            )
            decoded_key = tl.where(
                is_current[:, None] & valid_channel[None, :],
                current_key_row[None, :],
                decoded_key,
            )
            qk = tl.sum(decoded_key * query_row[None, :], axis=1) * scaling
            qk = tl.where(row_valid, qk, -float("inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, axis=0))
            p = tl.exp(qk - m_ij)
            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            accumulator *= alpha

            decoded_value = _load_cross_loop_v(
                value_payload0 + head_prefix * ((channels + 1) // 2),
                value_minimum0 + head_batch * prefix_tokens * value_channel_groups,
                value_scale0 + head_batch * prefix_tokens * value_channel_groups,
                prefix_token,
                channel,
                valid_prefix,
                prefix_tokens,
                channels,
                value_channel_groups,
                group_size,
            )
            if LOOP >= 1:
                decoded_value += _load_cross_loop_v(
                    value_payload1 + head_prefix * ((channels + 1) // 2),
                    value_minimum1 + head_batch * prefix_tokens * value_channel_groups,
                    value_scale1 + head_batch * prefix_tokens * value_channel_groups,
                    prefix_token,
                    channel,
                    valid_prefix,
                    prefix_tokens,
                    channels,
                    value_channel_groups,
                    group_size,
                )
            if LOOP >= 2 and HAS2:
                if BF16_ABSOLUTE_OVERRIDES:
                    update_value2 = tl.load(
                        value_absolute2
                        + (head_batch * tokens2 + safe2[:, None]) * channels
                        + channel[None, :],
                        mask=active2[:, None] & valid_channel[None, :],
                        other=0.0,
                    ).to(tl.float32)
                else:
                    update_value2 = _load_cross_loop_v(
                        value_payload2 + head_batch * tokens2 * ((channels + 1) // 2),
                        value_minimum2 + head_batch * tokens2 * value_channel_groups,
                        value_scale2 + head_batch * tokens2 * value_channel_groups,
                        safe2,
                        channel,
                        active2[:, None] & valid_channel[None, :],
                        tokens2,
                        channels,
                        value_channel_groups,
                        group_size,
                    )
                if ABSOLUTE_OVERRIDES:
                    decoded_value = tl.where(
                        active2[:, None], update_value2, decoded_value
                    )
                else:
                    decoded_value += update_value2
            if LOOP >= 3 and HAS3:
                if BF16_ABSOLUTE_OVERRIDES:
                    update_value3 = tl.load(
                        value_absolute3
                        + (head_batch * tokens3 + safe3[:, None]) * channels
                        + channel[None, :],
                        mask=active3[:, None] & valid_channel[None, :],
                        other=0.0,
                    ).to(tl.float32)
                else:
                    update_value3 = _load_cross_loop_v(
                        value_payload3 + head_batch * tokens3 * ((channels + 1) // 2),
                        value_minimum3 + head_batch * tokens3 * value_channel_groups,
                        value_scale3 + head_batch * tokens3 * value_channel_groups,
                        safe3,
                        channel,
                        active3[:, None] & valid_channel[None, :],
                        tokens3,
                        channels,
                        value_channel_groups,
                        group_size,
                    )
                if ABSOLUTE_OVERRIDES:
                    decoded_value = tl.where(
                        active3[:, None], update_value3, decoded_value
                    )
                else:
                    decoded_value += update_value3
            decoded_value = tl.where(
                is_tail[:, None] & valid_channel[None, :],
                tl.load(
                    tail_value + tail_offset,
                    mask=is_tail[:, None] & valid_channel[None, :],
                    other=0.0,
                ).to(tl.float32),
                decoded_value,
            )
            decoded_value = tl.where(
                is_current[:, None] & valid_channel[None, :],
                current_value_row[None, :],
                decoded_value,
            )
            accumulator += tl.sum(p[:, None] * decoded_value, axis=0)
            m_i = m_ij
            start += BLOCK_N

        if SPLIT_KV:
            partial = head_batch * num_splits + split
            scalar = tl.arange(0, 1)
            tl.store(partial_maximum + partial + scalar, m_i)
            tl.store(partial_denominator + partial + scalar, l_i)
            tl.store(
                partial_accumulator + partial * channels + channel,
                accumulator,
                mask=valid_channel,
            )
        else:
            target = accumulator / l_i
            source = tl.load(
                source_output + head_batch * channels + channel,
                mask=valid_channel,
                other=0.0,
            ).to(tl.float32)
            mass = tl.load(global_mass + head_batch).to(tl.float32)
            tl.store(
                output + head_batch * channels + channel,
                mass * (target - source),
                mask=valid_channel,
            )


    @triton.jit(do_not_specialize=_DENSE_SOURCE_DO_NOT_SPECIALIZE)
    def _cross_loop_dense_source_attention_kernel(
        query,
        current_key,
        current_value,
        key_payload0,
        key_minimum0,
        key_scale0,
        key_row_groups0,
        key_payload1,
        key_minimum1,
        key_scale1,
        key_row_groups1,
        value_payload0,
        value_minimum0,
        value_scale0,
        value_payload1,
        value_minimum1,
        value_scale1,
        tail_key,
        tail_value,
        output,
        logits,
        partial_maximum,
        partial_denominator,
        partial_accumulator,
        prefix_tokens,
        tail_tokens,
        tail_stride,
        num_splits,
        scaling,
        channels: tl.constexpr,
        key_groups0,
        key_groups1,
        value_channel_groups,
        group_size: tl.constexpr,
        LOOP: tl.constexpr,
        STORE_LOGITS: tl.constexpr,
        ROW_GROUPS0: tl.constexpr,
        ROW_GROUPS1: tl.constexpr,
        SPLIT_KV: tl.constexpr,
        SPLIT_N: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Online-softmax attention over packed Loop-1/2 K/V plus current row."""

        head_batch = tl.program_id(0)
        channel = tl.arange(0, BLOCK_D)
        valid_channel = channel < channels
        query_row = tl.load(
            query + head_batch * channels + channel,
            mask=valid_channel,
            other=0.0,
        ).to(tl.float32)
        current_key_row = tl.load(
            current_key + head_batch * channels + channel,
            mask=valid_channel,
            other=0.0,
        ).to(tl.float32)
        current_value_row = tl.load(
            current_value + head_batch * channels + channel,
            mask=valid_channel,
            other=0.0,
        ).to(tl.float32)

        past_tokens = prefix_tokens + tail_tokens
        total_rows = past_tokens + 1
        split = tl.program_id(1) if SPLIT_KV else 0
        m_i = tl.full((1,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((1,), dtype=tl.float32)
        accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
        start = split * SPLIT_N if SPLIT_KV else 0
        stop = tl.minimum(total_rows, start + SPLIT_N) if SPLIT_KV else total_rows
        while start < stop:
            row = start + tl.arange(0, BLOCK_N)
            valid_row = row < total_rows
            is_current = valid_row & (row == past_tokens)
            is_prefix = valid_row & (row < prefix_tokens)
            prefix_token = tl.where(is_prefix, row, 0)
            valid_prefix = is_prefix[:, None] & valid_channel[None, :]
            head_prefix = head_batch * prefix_tokens

            decoded_key = _load_cross_loop_k(
                key_payload0 + head_prefix * ((channels + 1) // 2),
                key_minimum0 + head_batch * key_groups0 * channels,
                key_scale0 + head_batch * key_groups0 * channels,
                key_row_groups0,
                prefix_token,
                channel,
                valid_prefix,
                prefix_tokens,
                channels,
                key_groups0,
                group_size,
                ROW_GROUPS0,
            )
            if LOOP >= 1:
                decoded_key += _load_cross_loop_k(
                    key_payload1 + head_prefix * ((channels + 1) // 2),
                    key_minimum1 + head_batch * key_groups1 * channels,
                    key_scale1 + head_batch * key_groups1 * channels,
                    key_row_groups1,
                    prefix_token,
                    channel,
                    valid_prefix,
                    prefix_tokens,
                    channels,
                    key_groups1,
                    group_size,
                    ROW_GROUPS1,
                )
            is_tail = valid_row & ~is_current & ~is_prefix
            tail_token = tl.where(is_tail, row - prefix_tokens, 0)
            tail_offset = (
                (head_batch * tail_stride + tail_token[:, None]) * channels
                + channel[None, :]
            )
            decoded_key = tl.where(
                is_tail[:, None] & valid_channel[None, :],
                tl.load(
                    tail_key + tail_offset,
                    mask=is_tail[:, None] & valid_channel[None, :],
                    other=0.0,
                ).to(tl.float32),
                decoded_key,
            )
            decoded_key = tl.where(
                is_current[:, None] & valid_channel[None, :],
                current_key_row[None, :],
                decoded_key,
            )
            qk = tl.sum(decoded_key * query_row[None, :], axis=1) * scaling
            qk = tl.where(valid_row, qk, -float("inf"))
            if STORE_LOGITS:
                tl.store(
                    logits + head_batch * total_rows + row,
                    qk,
                    mask=valid_row,
                )
            m_ij = tl.maximum(m_i, tl.max(qk, axis=0))
            p = tl.exp(qk - m_ij)
            alpha = tl.exp(m_i - m_ij)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            accumulator *= alpha

            decoded_value = _load_cross_loop_v(
                value_payload0 + head_prefix * ((channels + 1) // 2),
                value_minimum0
                + head_batch * prefix_tokens * value_channel_groups,
                value_scale0
                + head_batch * prefix_tokens * value_channel_groups,
                prefix_token,
                channel,
                valid_prefix,
                prefix_tokens,
                channels,
                value_channel_groups,
                group_size,
            )
            if LOOP >= 1:
                decoded_value += _load_cross_loop_v(
                    value_payload1 + head_prefix * ((channels + 1) // 2),
                    value_minimum1
                    + head_batch * prefix_tokens * value_channel_groups,
                    value_scale1
                    + head_batch * prefix_tokens * value_channel_groups,
                    prefix_token,
                    channel,
                    valid_prefix,
                    prefix_tokens,
                    channels,
                    value_channel_groups,
                    group_size,
                )
            decoded_value = tl.where(
                is_tail[:, None] & valid_channel[None, :],
                tl.load(
                    tail_value + tail_offset,
                    mask=is_tail[:, None] & valid_channel[None, :],
                    other=0.0,
                ).to(tl.float32),
                decoded_value,
            )
            decoded_value = tl.where(
                is_current[:, None] & valid_channel[None, :],
                current_value_row[None, :],
                decoded_value,
            )
            accumulator += tl.sum(p[:, None] * decoded_value, axis=0)
            m_i = m_ij
            start += BLOCK_N

        if SPLIT_KV:
            partial = head_batch * num_splits + split
            scalar = tl.arange(0, 1)
            tl.store(partial_maximum + partial + scalar, m_i)
            tl.store(partial_denominator + partial + scalar, l_i)
            tl.store(
                partial_accumulator + partial * channels + channel,
                accumulator,
                mask=valid_channel,
            )
        else:
            tl.store(
                output + head_batch * channels + channel,
                accumulator / l_i,
                mask=valid_channel,
            )


    @triton.jit
    def _merge_split_kv_attention_kernel(
        partial_maximum,
        partial_denominator,
        partial_accumulator,
        source_output,
        global_mass,
        output,
        num_splits,
        channels: tl.constexpr,
        APPLY_CORRECTION: tl.constexpr,
        BLOCK_S: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Numerically stable merge of independent online-softmax KV tiles."""

        head_batch = tl.program_id(0)
        channel_block = tl.program_id(1)
        split = tl.arange(0, BLOCK_S)
        valid_split = split < num_splits
        base = head_batch * num_splits
        maxima = tl.load(
            partial_maximum + base + split,
            mask=valid_split,
            other=-float("inf"),
        ).to(tl.float32)
        maximum = tl.max(maxima, axis=0)
        rescale = tl.exp(maxima - maximum)
        denominators = tl.load(
            partial_denominator + base + split,
            mask=valid_split,
            other=0.0,
        ).to(tl.float32)
        denominator = tl.sum(denominators * rescale, axis=0)

        channel = channel_block * BLOCK_D + tl.arange(0, BLOCK_D)
        valid_channel = channel < channels
        offsets = (
            (base + split[:, None]) * channels + channel[None, :]
        )
        partials = tl.load(
            partial_accumulator + offsets,
            mask=valid_split[:, None] & valid_channel[None, :],
            other=0.0,
        ).to(tl.float32)
        numerator = tl.sum(partials * rescale[:, None], axis=0)
        target = numerator / denominator
        if APPLY_CORRECTION:
            source = tl.load(
                source_output + head_batch * channels + channel,
                mask=valid_channel,
                other=0.0,
            ).to(tl.float32)
            mass = tl.load(global_mass + head_batch).to(tl.float32)
            target = mass * (target - source)
        tl.store(
            output + head_batch * channels + channel,
            target,
            mask=valid_channel,
        )


def _torch_pack_4bit(values: torch.Tensor, *, axis: str, group_size: int) -> Packed4Bit:
    if axis == "per_channel":
        return pack_k_per_channel(values, group_size=group_size)
    if axis == "per_token":
        return pack_v_per_token(values, group_size=group_size)
    raise ValueError("axis must be 'per_channel' or 'per_token'")


def _require_production_specialization(values: torch.Tensor, group_size: int) -> None:
    if values.ndim != 4:
        raise ValueError("values must have shape [batch, heads, tokens, channels]")
    if int(values.shape[-1]) != 128:
        raise ValueError("the first CUDA kernel specializes head_dim=128")
    if int(group_size) != 64:
        raise ValueError("the first CUDA kernel specializes group_size=64")
    if values.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("the CUDA packing kernel requires FP16 or BF16 input")


def triton_pack_4bit(
    values: torch.Tensor,
    *,
    axis: str,
    group_size: int = 64,
    force_torch: bool = False,
) -> Packed4Bit:
    """Pack a regular K or V stream with physical uint8 nibbles."""

    if force_torch or not values.is_cuda or not _TRITON_AVAILABLE:
        return _torch_pack_4bit(values, axis=axis, group_size=group_size)
    _require_production_specialization(values, group_size)
    if axis not in {"per_channel", "per_token"}:
        raise ValueError("axis must be 'per_channel' or 'per_token'")
    values = values.contiguous()
    batch, heads, tokens, channels = (int(value) for value in values.shape)
    token_groups = triton.cdiv(tokens, group_size)
    channel_groups = triton.cdiv(channels, group_size)
    if axis == "per_channel":
        metadata_shape = (batch, heads, token_groups, channels)
    else:
        metadata_shape = (batch, heads, tokens, channel_groups)
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

    payload = torch.empty(
        (batch, heads, tokens, (channels + 1) // 2),
        device=values.device,
        dtype=torch.uint8,
    )
    total_bytes = payload.numel()
    _pack_nibbles_kernel[(triton.cdiv(total_bytes, 256),)](
        values,
        work_minimum,
        work_scale,
        payload,
        total_bytes,
        tokens=tokens,
        channels=channels,
        token_groups=token_groups,
        channel_groups=channel_groups,
        group_size=group_size,
        PER_CHANNEL=axis == "per_channel",
        BLOCK_BYTES=256,
        num_warps=4,
    )
    return Packed4Bit(
        payload=payload,
        minimum=minimum,
        scale=scale,
        original_shape=(batch, heads, tokens, channels),
        original_dtype=values.dtype,
        axis=axis,
        group_size=group_size,
    )


def _normalize_indices(packed: Packed4Bit, indices: torch.Tensor) -> tuple[torch.Tensor, bool, int]:
    if indices.ndim == 1:
        normalized = indices.to(device=packed.payload.device, dtype=torch.int32).contiguous()
        return normalized, True, int(normalized.numel())
    if indices.ndim == 4 and indices.shape[2] == 1:
        indices = indices.squeeze(2)
    if indices.ndim != 3 or tuple(indices.shape[:2]) != packed.original_shape[:2]:
        raise ValueError("indices must have shape [selected] or [batch, heads, selected]")
    normalized = indices.to(device=packed.payload.device, dtype=torch.int32).contiguous()
    return normalized, False, int(normalized.shape[-1])


def triton_gather_dequantize(
    packed: Packed4Bit,
    indices: torch.Tensor,
    *,
    force_torch: bool = False,
    validate_indices: bool = True,
) -> torch.Tensor:
    """Decode only selected token rows from a physical 4-bit stream."""

    normalized, shared_indices, selected_rows = _normalize_indices(packed, indices)
    tokens = packed.original_shape[2]
    if validate_indices and normalized.numel() and (
        int(normalized.min().item()) < 0 or int(normalized.max().item()) >= tokens
    ):
        raise IndexError("decode row index is outside the packed token range")
    if force_torch or not packed.payload.is_cuda or not _TRITON_AVAILABLE:
        if not shared_indices:
            decoded = packed.decode()
            output = torch.empty(
                (*packed.original_shape[:2], selected_rows, packed.original_shape[3]),
                device=packed.payload.device,
                dtype=packed.original_dtype,
            )
            for batch in range(packed.original_shape[0]):
                for head in range(packed.original_shape[1]):
                    output[batch, head] = decoded[batch, head].index_select(
                        0,
                        normalized[batch, head].long(),
                    )
            return output
        return packed.decode_rows(normalized.long())

    batch, heads, _tokens, channels = packed.original_shape
    if channels != 128 or packed.group_size not in (1, 64):
        raise ValueError("the first CUDA reader specializes head_dim=128 and group_size=64")
    output = torch.empty(
        (batch, heads, selected_rows, channels),
        device=packed.payload.device,
        dtype=packed.original_dtype,
    )
    token_groups = int(packed.minimum.shape[2]) if packed.axis == "per_channel" else 1
    channel_groups = int(packed.minimum.shape[3]) if packed.axis == "per_token" else 1
    row_groups = packed.row_groups if packed.row_groups is not None else normalized
    _gather_dequantize_kernel[(batch * heads * selected_rows, triton.cdiv(channels, 128))](
        packed.payload,
        packed.minimum,
        packed.scale,
        row_groups,
        normalized,
        output,
        selected_rows=selected_rows,
        tokens=tokens,
        channels=channels,
        token_groups=token_groups,
        channel_groups=channel_groups,
        group_size=packed.group_size,
        SHARED_INDICES=shared_indices,
        PER_CHANNEL=packed.axis == "per_channel",
        HAS_ROW_GROUPS=packed.row_groups is not None,
        BLOCK_CHANNELS=128,
        num_warps=4,
    )
    return output


def _gather_reference(packed: Packed4Bit, normalized: torch.Tensor, shared: bool) -> torch.Tensor:
    if shared:
        return packed.decode_rows(normalized.long()).float()
    decoded = packed.decode()
    return torch.gather(
        decoded,
        2,
        normalized.long().unsqueeze(-1).expand(-1, -1, -1, packed.original_shape[-1]),
    ).float()


def triton_packed_qk(
    query: torch.Tensor,
    packed: Packed4Bit,
    indices: torch.Tensor,
    *,
    force_torch: bool = False,
) -> torch.Tensor:
    """Multiply one query by selected packed per-channel K rows directly."""

    normalized, shared_indices, selected_rows = _normalize_indices(packed, indices)
    if packed.axis != "per_channel":
        raise ValueError("packed QK requires per-channel K storage")
    if query.ndim != 4 or query.shape[-2] != 1:
        raise ValueError("query must have shape [batch, heads, 1, head_dim]")
    if tuple(query.shape[:2]) != packed.original_shape[:2] or query.shape[-1] != packed.original_shape[-1]:
        raise ValueError("query shape differs from packed K")
    if force_torch or not query.is_cuda or not _TRITON_AVAILABLE:
        gathered = _gather_reference(packed, normalized, shared_indices)
        return torch.matmul(query.float(), gathered.transpose(-1, -2))
    if packed.original_shape[-1] != 128 or packed.group_size not in (1, 64):
        raise ValueError("the first packed QK kernel specializes head_dim=128")
    query = query.contiguous()
    output = torch.empty(
        (*query.shape[:2], 1, selected_rows),
        device=query.device,
        dtype=torch.float32,
    )
    token_groups = int(packed.minimum.shape[2])
    row_groups = packed.row_groups if packed.row_groups is not None else normalized
    _packed_qk_kernel[(query.shape[0] * query.shape[1], triton.cdiv(selected_rows, 64))](
        query,
        packed.payload,
        packed.minimum,
        packed.scale,
        row_groups,
        normalized,
        output,
        selected_rows=selected_rows,
        tokens=packed.original_shape[2],
        channels=packed.original_shape[3],
        token_groups=token_groups,
        group_size=packed.group_size,
        SHARED_INDICES=shared_indices,
        HAS_ROW_GROUPS=packed.row_groups is not None,
        BLOCK_N=64,
        BLOCK_D=128,
        num_warps=4,
    )
    return output


def triton_packed_pv(
    weights: torch.Tensor,
    packed: Packed4Bit,
    indices: torch.Tensor,
    *,
    force_torch: bool = False,
) -> torch.Tensor:
    """Multiply attention weights by selected packed per-token V rows directly."""

    normalized, shared_indices, selected_rows = _normalize_indices(packed, indices)
    if packed.axis != "per_token":
        raise ValueError("packed PV requires per-token V storage")
    expected = (*packed.original_shape[:2], 1, selected_rows)
    if tuple(weights.shape) != expected:
        raise ValueError("weights must have shape [batch, heads, 1, selected]")
    if force_torch or not weights.is_cuda or not _TRITON_AVAILABLE:
        gathered = _gather_reference(packed, normalized, shared_indices)
        return torch.matmul(weights.float(), gathered)
    if packed.original_shape[-1] != 128 or packed.group_size != 64:
        raise ValueError("the first packed PV kernel specializes head_dim=128 and group_size=64")
    weights = weights.contiguous()
    output = torch.empty(
        (*weights.shape[:2], 1, packed.original_shape[-1]),
        device=weights.device,
        dtype=torch.float32,
    )
    _packed_pv_kernel[(weights.shape[0] * weights.shape[1],)](
        weights,
        packed.payload,
        packed.minimum,
        packed.scale,
        normalized,
        output,
        selected_rows=selected_rows,
        tokens=packed.original_shape[2],
        channels=packed.original_shape[3],
        channel_groups=int(packed.minimum.shape[3]),
        group_size=packed.group_size,
        SHARED_INDICES=shared_indices,
        BLOCK_N=64,
        BLOCK_D=128,
        num_warps=4,
    )
    return output


def _cross_loop_stream(
    streams: tuple[Packed4Bit | None, ...],
    index: int,
) -> tuple[Packed4Bit, bool]:
    anchor = streams[0]
    if anchor is None:
        raise ValueError("cross-loop packed storage requires an anchor stream")
    stream = streams[index]
    return (stream if stream is not None else anchor), stream is not None


def _row_groups_or_dummy(packed: Packed4Bit, dummy: torch.Tensor) -> torch.Tensor:
    return packed.row_groups if packed.row_groups is not None else dummy


def _native_absolute_streams(
    streams: tuple[torch.Tensor | None, ...] | None,
    *,
    anchor: Packed4Bit,
    dummy: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[bool, bool], tuple[int, int]]:
    if streams is None:
        streams = (None, None, None, None)
    if len(streams) != 4:
        raise ValueError("four absolute override stream slots are required")
    selected = (streams[2], streams[3])
    present = tuple(stream is not None for stream in selected)
    for stream in selected:
        if stream is None:
            continue
        if (
            stream.dtype != torch.bfloat16
            or stream.device != dummy.device
            or tuple(stream.shape[:2]) != anchor.original_shape[:2]
            or int(stream.shape[3]) != anchor.original_shape[3]
        ):
            raise ValueError("native absolute overrides must be contiguous BF16 cache rows")
        if not stream.is_contiguous():
            raise ValueError("native absolute override streams must be contiguous")
    pointers = tuple(stream if stream is not None else dummy for stream in selected)
    tokens = tuple(int(stream.shape[2]) if stream is not None else 0 for stream in selected)
    return pointers, present, tokens  # type: ignore[return-value]


def triton_cross_loop_qk(
    query: torch.Tensor,
    packed_keys: tuple[Packed4Bit | None, ...],
    rank_maps: tuple[torch.Tensor, torch.Tensor],
    tail_key: torch.Tensor,
    indices: torch.Tensor,
    *,
    loop: int,
    absolute_keys: tuple[torch.Tensor | None, ...] | None = None,
    absolute_late_overrides: bool = False,
) -> torch.Tensor:
    """Fuse anchor, recurrent K deltas, and the BF16 tail into one QK launch."""

    if not _TRITON_AVAILABLE or not query.is_cuda:
        raise RuntimeError("cross-loop QK fusion requires Triton CUDA")
    if len(packed_keys) != 4 or loop not in range(4):
        raise ValueError("four packed streams and loop in [0, 3] are required")
    anchor, _ = _cross_loop_stream(packed_keys, 0)
    normalized, shared, selected = _normalize_indices(anchor, indices)
    if shared:
        normalized = normalized[None, None].expand(*anchor.original_shape[:2], -1).contiguous()
    if tuple(query.shape) != (*anchor.original_shape[:2], 1, anchor.original_shape[3]):
        raise ValueError("query shape differs from cross-loop K storage")
    if tuple(tail_key.shape[:2]) != anchor.original_shape[:2] or tail_key.shape[-1] != anchor.original_shape[3]:
        raise ValueError("tail K shape differs from packed storage")
    streams = tuple(_cross_loop_stream(packed_keys, index) for index in range(4))
    packed = tuple(item[0] for item in streams)
    packed_present = tuple(item[1] for item in streams)
    absolute, absolute_present, absolute_tokens = _native_absolute_streams(
        absolute_keys,
        anchor=anchor,
        dummy=tail_key,
    )
    if any(absolute_present) and not absolute_late_overrides:
        raise ValueError("BF16 override streams require absolute_late_overrides=True")
    if any(absolute_present) and any(packed_present[2:]):
        raise ValueError("late override streams cannot mix packed and native BF16 storage")
    present = (
        packed_present[0],
        packed_present[1],
        packed_present[2] or absolute_present[0],
        packed_present[3] or absolute_present[1],
    )
    dummy_groups = rank_maps[0]
    output = torch.empty(
        (*anchor.original_shape[:2], 1, selected),
        device=query.device,
        dtype=torch.float32,
    )
    _cross_loop_qk_kernel[
        (query.shape[0] * query.shape[1], triton.cdiv(selected, 64))
    ](
        query.contiguous(),
        normalized,
        packed[0].payload,
        packed[0].minimum,
        packed[0].scale,
        _row_groups_or_dummy(packed[0], dummy_groups),
        packed[1].payload,
        packed[1].minimum,
        packed[1].scale,
        _row_groups_or_dummy(packed[1], dummy_groups),
        packed[2].payload,
        packed[2].minimum,
        packed[2].scale,
        _row_groups_or_dummy(packed[2], dummy_groups),
        packed[3].payload,
        packed[3].minimum,
        packed[3].scale,
        _row_groups_or_dummy(packed[3], dummy_groups),
        absolute[0],
        absolute[1],
        rank_maps[0],
        rank_maps[1],
        tail_key.contiguous(),
        output,
        selected_rows=selected,
        prefix_tokens=anchor.original_shape[2],
        tail_tokens=tail_key.shape[2],
        channels=anchor.original_shape[3],
        groups0=packed[0].minimum.shape[2],
        groups1=packed[1].minimum.shape[2],
        groups2=packed[2].minimum.shape[2],
        groups3=packed[3].minimum.shape[2],
        tokens2=absolute_tokens[0] if absolute_present[0] else packed[2].original_shape[2],
        tokens3=absolute_tokens[1] if absolute_present[1] else packed[3].original_shape[2],
        group_size=anchor.group_size,
        LOOP=loop,
        HAS2=present[2],
        HAS3=present[3],
        ABSOLUTE_OVERRIDES=bool(absolute_late_overrides),
        BF16_ABSOLUTE_OVERRIDES=any(absolute_present),
        ROW_GROUPS0=packed[0].row_groups is not None,
        ROW_GROUPS1=packed[1].row_groups is not None,
        ROW_GROUPS2=packed[2].row_groups is not None,
        ROW_GROUPS3=packed[3].row_groups is not None,
        BLOCK_N=64,
        BLOCK_D=128,
        num_warps=4,
    )
    return output


def triton_cross_loop_pv(
    weights: torch.Tensor,
    packed_values: tuple[Packed4Bit | None, ...],
    rank_maps: tuple[torch.Tensor, torch.Tensor],
    tail_value: torch.Tensor,
    indices: torch.Tensor,
    *,
    loop: int,
    absolute_values: tuple[torch.Tensor | None, ...] | None = None,
    absolute_late_overrides: bool = False,
    block_channels: int = 4,
    num_warps: int = 4,
) -> torch.Tensor:
    """Fuse anchor, recurrent V deltas, and the BF16 tail into one PV launch."""

    if not _TRITON_AVAILABLE or not weights.is_cuda:
        raise RuntimeError("cross-loop PV fusion requires Triton CUDA")
    if len(packed_values) != 4 or loop not in range(4):
        raise ValueError("four packed streams and loop in [0, 3] are required")
    anchor, _ = _cross_loop_stream(packed_values, 0)
    normalized, shared, selected = _normalize_indices(anchor, indices)
    if shared:
        normalized = normalized[None, None].expand(*anchor.original_shape[:2], -1).contiguous()
    expected = (*anchor.original_shape[:2], 1, selected)
    if tuple(weights.shape) != expected:
        raise ValueError("weights shape differs from cross-loop V storage")
    if tuple(tail_value.shape[:2]) != anchor.original_shape[:2] or tail_value.shape[-1] != anchor.original_shape[3]:
        raise ValueError("tail V shape differs from packed storage")
    streams = tuple(_cross_loop_stream(packed_values, index) for index in range(4))
    packed = tuple(item[0] for item in streams)
    packed_present = tuple(item[1] for item in streams)
    absolute, absolute_present, absolute_tokens = _native_absolute_streams(
        absolute_values,
        anchor=anchor,
        dummy=tail_value,
    )
    if any(absolute_present) and not absolute_late_overrides:
        raise ValueError("BF16 override streams require absolute_late_overrides=True")
    if any(absolute_present) and any(packed_present[2:]):
        raise ValueError("late override streams cannot mix packed and native BF16 storage")
    present = (
        packed_present[0],
        packed_present[1],
        packed_present[2] or absolute_present[0],
        packed_present[3] or absolute_present[1],
    )
    output = torch.empty(
        (*anchor.original_shape[:2], 1, anchor.original_shape[3]),
        device=weights.device,
        dtype=torch.float32,
    )
    block_channels, num_warps = _validate_cross_loop_pv_tiling(
        anchor.original_shape[3], block_channels, num_warps
    )
    _cross_loop_pv_kernel[
        _cross_loop_pv_grid(
            weights.shape[0] * weights.shape[1],
            anchor.original_shape[3],
            block_channels=block_channels,
        )
    ](
        weights.float().contiguous(),
        normalized,
        packed[0].payload,
        packed[0].minimum,
        packed[0].scale,
        packed[1].payload,
        packed[1].minimum,
        packed[1].scale,
        packed[2].payload,
        packed[2].minimum,
        packed[2].scale,
        packed[3].payload,
        packed[3].minimum,
        packed[3].scale,
        absolute[0],
        absolute[1],
        rank_maps[0],
        rank_maps[1],
        tail_value.contiguous(),
        output,
        selected_rows=selected,
        prefix_tokens=anchor.original_shape[2],
        tail_tokens=tail_value.shape[2],
        channels=anchor.original_shape[3],
        tokens2=absolute_tokens[0] if absolute_present[0] else packed[2].original_shape[2],
        tokens3=absolute_tokens[1] if absolute_present[1] else packed[3].original_shape[2],
        channel_groups=anchor.minimum.shape[3],
        group_size=anchor.group_size,
        LOOP=loop,
        HAS2=present[2],
        HAS3=present[3],
        ABSOLUTE_OVERRIDES=bool(absolute_late_overrides),
        BF16_ABSOLUTE_OVERRIDES=any(absolute_present),
        BLOCK_N=64,
        BLOCK_D=block_channels,
        num_warps=num_warps,
    )
    return output


def triton_cross_loop_sparse_attention_delta(
    *,
    query: torch.Tensor,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    packed_keys: tuple[Packed4Bit | None, ...],
    packed_values: tuple[Packed4Bit | None, ...],
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
    absolute_keys: tuple[torch.Tensor | None, ...] | None = None,
    absolute_values: tuple[torch.Tensor | None, ...] | None = None,
    absolute_late_overrides: bool = False,
    split_kv: bool = False,
    split_rows: int = 128,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    block_rows: int = 64,
    num_warps: int = 8,
) -> torch.Tensor:
    """Fuse packed QK, subset softmax, packed PV, and cached-mass correction."""

    if not _TRITON_AVAILABLE or not query.is_cuda:
        raise RuntimeError("cross-loop sparse attention fusion requires Triton CUDA")
    if len(packed_keys) != 4 or len(packed_values) != 4 or loop not in (2, 3):
        raise ValueError("four packed K/V streams and Loop 3 or 4 are required")
    key_anchor, _ = _cross_loop_stream(packed_keys, 0)
    value_anchor, _ = _cross_loop_stream(packed_values, 0)
    normalized, shared, selected = _normalize_indices(key_anchor, indices)
    if shared:
        normalized = normalized[None, None].expand(
            *key_anchor.original_shape[:2], -1
        ).contiguous()
    expected_vector = (*key_anchor.original_shape[:2], 1, key_anchor.original_shape[3])
    if any(
        tuple(tensor.shape) != expected_vector
        for tensor in (
            query,
            current_key,
            current_value,
            source_selected_output,
        )
    ):
        raise ValueError("attention vectors differ from packed K/V storage")
    if tuple(valid.shape) != tuple(normalized.shape):
        raise ValueError("valid mask must match normalized selected indices")
    if tuple(global_mass.shape) != (*key_anchor.original_shape[:2], 1, 1):
        raise ValueError("global mass must have shape [batch, heads, 1, 1]")
    if key_anchor.original_shape != value_anchor.original_shape:
        raise ValueError("packed K/V anchors must have identical shapes")
    if key_anchor.original_shape[-1] != 128 or key_anchor.group_size != 64:
        raise ValueError("the fused kernel specializes head_dim=128 and group_size=64")
    if tuple(tail_key.shape) != tuple(tail_value.shape):
        raise ValueError("tail K/V tensors must have identical shapes")

    key_streams = tuple(_cross_loop_stream(packed_keys, index) for index in range(4))
    value_streams = tuple(
        _cross_loop_stream(packed_values, index) for index in range(4)
    )
    packed_key = tuple(item[0] for item in key_streams)
    packed_value = tuple(item[0] for item in value_streams)
    packed_key_present = tuple(item[1] for item in key_streams)
    packed_value_present = tuple(item[1] for item in value_streams)
    key_absolute, key_absolute_present, key_absolute_tokens = _native_absolute_streams(
        absolute_keys,
        anchor=key_anchor,
        dummy=tail_key,
    )
    value_absolute, value_absolute_present, value_absolute_tokens = (
        _native_absolute_streams(
            absolute_values,
            anchor=value_anchor,
            dummy=tail_value,
        )
    )
    if key_absolute_present != value_absolute_present:
        raise ValueError("native absolute K/V streams must have matching presence")
    if key_absolute_tokens != value_absolute_tokens:
        raise ValueError("native absolute K/V streams must have matching row counts")
    if any(key_absolute_present) and not absolute_late_overrides:
        raise ValueError("BF16 override streams require absolute_late_overrides=True")
    if any(key_absolute_present) and (
        any(packed_key_present[2:]) or any(packed_value_present[2:])
    ):
        raise ValueError("late override streams cannot mix packed and native BF16 storage")
    key_present = (
        packed_key_present[0],
        packed_key_present[1],
        packed_key_present[2] or key_absolute_present[0],
        packed_key_present[3] or key_absolute_present[1],
    )
    value_present = (
        packed_value_present[0],
        packed_value_present[1],
        packed_value_present[2] or value_absolute_present[0],
        packed_value_present[3] or value_absolute_present[1],
    )
    if key_present != value_present:
        raise ValueError("cross-loop K/V delta streams must have matching presence")

    output = torch.empty(expected_vector, device=query.device, dtype=torch.float32)
    split_kv = bool(split_kv)
    block_rows = int(block_rows)
    num_warps = int(num_warps)
    if block_rows not in (32, 64, 128, 256):
        raise ValueError("block_rows must be one of 32, 64, 128, or 256")
    if num_warps not in (2, 4, 8):
        raise ValueError("num_warps must be 2, 4, or 8")
    if split_kv:
        num_splits, merge_width = _split_kv_geometry(
            selected, split_rows=int(split_rows)
        )
        expected_workspace = (
            (query.shape[0] * query.shape[1], num_splits),
            (query.shape[0] * query.shape[1], num_splits),
            (
                query.shape[0] * query.shape[1],
                num_splits,
                key_anchor.original_shape[3],
            ),
        )
        if workspace is None:
            partial_maximum = torch.empty(
                expected_workspace[0], device=query.device, dtype=torch.float32
            )
            partial_denominator = torch.empty_like(partial_maximum)
            partial_accumulator = torch.empty(
                expected_workspace[2], device=query.device, dtype=torch.float32
            )
        else:
            if tuple(tensor.shape for tensor in workspace) != expected_workspace:
                raise ValueError("split-KV workspace shape does not match sparse attention")
            partial_maximum, partial_denominator, partial_accumulator = workspace
    else:
        num_splits, merge_width = 1, 1
        partial_maximum = output
        partial_denominator = output
        partial_accumulator = output
    dummy_groups = rank_maps[0]
    _cross_loop_sparse_attention_delta_kernel[
        (query.shape[0] * query.shape[1], num_splits)
    ](
        query.contiguous(),
        current_key.contiguous(),
        current_value.contiguous(),
        normalized,
        valid.to(device=query.device, dtype=torch.bool).contiguous(),
        packed_key[0].payload,
        packed_key[0].minimum,
        packed_key[0].scale,
        _row_groups_or_dummy(packed_key[0], dummy_groups),
        packed_key[1].payload,
        packed_key[1].minimum,
        packed_key[1].scale,
        _row_groups_or_dummy(packed_key[1], dummy_groups),
        packed_key[2].payload,
        packed_key[2].minimum,
        packed_key[2].scale,
        _row_groups_or_dummy(packed_key[2], dummy_groups),
        packed_key[3].payload,
        packed_key[3].minimum,
        packed_key[3].scale,
        _row_groups_or_dummy(packed_key[3], dummy_groups),
        key_absolute[0],
        key_absolute[1],
        packed_value[0].payload,
        packed_value[0].minimum,
        packed_value[0].scale,
        packed_value[1].payload,
        packed_value[1].minimum,
        packed_value[1].scale,
        packed_value[2].payload,
        packed_value[2].minimum,
        packed_value[2].scale,
        packed_value[3].payload,
        packed_value[3].minimum,
        packed_value[3].scale,
        value_absolute[0],
        value_absolute[1],
        rank_maps[0],
        rank_maps[1],
        tail_key.contiguous(),
        tail_value.contiguous(),
        source_selected_output.contiguous(),
        global_mass.float().contiguous(),
        output,
        partial_maximum,
        partial_denominator,
        partial_accumulator,
        selected_rows=selected,
        prefix_tokens=key_anchor.original_shape[2],
        tail_tokens=tail_key.shape[2],
        current_position=int(current_position),
        num_splits=num_splits,
        scaling=float(scaling),
        channels=key_anchor.original_shape[3],
        key_groups0=packed_key[0].minimum.shape[2],
        key_groups1=packed_key[1].minimum.shape[2],
        key_groups2=packed_key[2].minimum.shape[2],
        key_groups3=packed_key[3].minimum.shape[2],
        tokens2=(
            key_absolute_tokens[0]
            if key_absolute_present[0]
            else packed_key[2].original_shape[2]
        ),
        tokens3=(
            key_absolute_tokens[1]
            if key_absolute_present[1]
            else packed_key[3].original_shape[2]
        ),
        value_channel_groups=value_anchor.minimum.shape[3],
        group_size=key_anchor.group_size,
        LOOP=loop,
        HAS2=key_present[2],
        HAS3=key_present[3],
        ABSOLUTE_OVERRIDES=bool(absolute_late_overrides),
        BF16_ABSOLUTE_OVERRIDES=any(key_absolute_present),
        ROW_GROUPS0=packed_key[0].row_groups is not None,
        ROW_GROUPS1=packed_key[1].row_groups is not None,
        ROW_GROUPS2=packed_key[2].row_groups is not None,
        ROW_GROUPS3=packed_key[3].row_groups is not None,
        SPLIT_KV=split_kv,
        SPLIT_N=int(split_rows),
        BLOCK_N=block_rows,
        BLOCK_D=128,
        num_warps=num_warps,
        num_stages=2,
    )
    if split_kv:
        _merge_split_kv_attention_kernel[
            (
                query.shape[0] * query.shape[1],
                triton.cdiv(key_anchor.original_shape[3], 16),
            )
        ](
            partial_maximum,
            partial_denominator,
            partial_accumulator,
            source_selected_output.contiguous(),
            global_mass.float().contiguous(),
            output,
            num_splits=num_splits,
            channels=key_anchor.original_shape[3],
            APPLY_CORRECTION=True,
            BLOCK_S=merge_width,
            BLOCK_D=16,
            num_warps=4,
        )
    return output


def triton_cross_loop_dense_source_attention(
    *,
    query: torch.Tensor,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    packed_keys: tuple[Packed4Bit | None, ...],
    packed_values: tuple[Packed4Bit | None, ...],
    tail_key: torch.Tensor,
    tail_value: torch.Tensor,
    tail_tokens: int,
    scaling: float,
    loop: int,
    return_logits: bool,
    split_kv: bool = False,
    split_rows: int = 128,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fuse packed Loop-1/2 QK, online softmax, and PV.

    ``split_kv`` gives single-query long-context decode token-parallel stage-1
    programs, followed by a stable online-softmax merge.  Short contexts keep
    the original one-program-per-head path.
    """

    if not _TRITON_AVAILABLE or not query.is_cuda:
        raise RuntimeError("packed dense source attention requires Triton CUDA")
    if len(packed_keys) != 4 or len(packed_values) != 4 or loop not in (0, 1):
        raise ValueError("four packed streams and Loop 1 or 2 are required")
    key0, _ = _cross_loop_stream(packed_keys, 0)
    key1, key1_present = _cross_loop_stream(packed_keys, 1)
    value0, _ = _cross_loop_stream(packed_values, 0)
    value1, value1_present = _cross_loop_stream(packed_values, 1)
    if loop == 1 and (not key1_present or not value1_present):
        raise ValueError("Loop 2 requires packed K/V delta streams")
    expected = (*key0.original_shape[:2], 1, key0.original_shape[3])
    if any(
        tuple(tensor.shape) != expected
        for tensor in (query, current_key, current_value)
    ):
        raise ValueError("attention vectors differ from packed K/V storage")
    if key0.original_shape != value0.original_shape:
        raise ValueError("packed K/V anchors must have identical shapes")
    if key0.original_shape[-1] != 128 or key0.group_size != 64:
        raise ValueError(
            "dense source kernel specializes head_dim=128 and group_size=64"
        )
    if tuple(tail_key.shape) != tuple(tail_value.shape):
        raise ValueError("tail K/V tensors must have identical shapes")
    tail_tokens = int(tail_tokens)
    if tail_tokens < 0 or tail_tokens > int(tail_key.shape[2]):
        raise ValueError("logical tail length exceeds tail storage")

    batch, heads, prefix_tokens, channels = key0.original_shape
    total_rows = prefix_tokens + tail_tokens + 1
    output = torch.empty(
        (batch, heads, 1, channels), device=query.device, dtype=torch.float32
    )
    logits = (
        torch.empty(
            (batch, heads, 1, total_rows),
            device=query.device,
            dtype=torch.float32,
        )
        if return_logits
        else torch.empty((1,), device=query.device, dtype=torch.float32)
    )
    dummy_groups = torch.empty((1,), device=query.device, dtype=torch.int32)
    split_kv = bool(split_kv)
    if split_kv:
        num_splits, merge_width = _split_kv_geometry(
            total_rows, split_rows=int(split_rows)
        )
        expected_workspace = (
            (batch * heads, num_splits),
            (batch * heads, num_splits),
            (batch * heads, num_splits, channels),
        )
        if workspace is None:
            partial_maximum = torch.empty(
                expected_workspace[0], device=query.device, dtype=torch.float32
            )
            partial_denominator = torch.empty_like(partial_maximum)
            partial_accumulator = torch.empty(
                expected_workspace[2], device=query.device, dtype=torch.float32
            )
        else:
            if tuple(tensor.shape for tensor in workspace) != expected_workspace:
                raise ValueError("split-KV workspace shape does not match dense attention")
            partial_maximum, partial_denominator, partial_accumulator = workspace
    else:
        num_splits, merge_width = 1, 1
        partial_maximum = output
        partial_denominator = output
        partial_accumulator = output
    _cross_loop_dense_source_attention_kernel[
        (batch * heads, num_splits)
    ](
        query.contiguous(),
        current_key.contiguous(),
        current_value.contiguous(),
        key0.payload,
        key0.minimum,
        key0.scale,
        _row_groups_or_dummy(key0, dummy_groups),
        key1.payload,
        key1.minimum,
        key1.scale,
        _row_groups_or_dummy(key1, dummy_groups),
        value0.payload,
        value0.minimum,
        value0.scale,
        value1.payload,
        value1.minimum,
        value1.scale,
        tail_key.contiguous(),
        tail_value.contiguous(),
        output,
        logits,
        partial_maximum,
        partial_denominator,
        partial_accumulator,
        prefix_tokens=prefix_tokens,
        tail_tokens=tail_tokens,
        tail_stride=int(tail_key.shape[2]),
        num_splits=num_splits,
        scaling=float(scaling),
        channels=channels,
        key_groups0=int(key0.minimum.shape[2]),
        key_groups1=int(key1.minimum.shape[2]),
        value_channel_groups=int(value0.minimum.shape[3]),
        group_size=key0.group_size,
        LOOP=int(loop),
        STORE_LOGITS=bool(return_logits),
        ROW_GROUPS0=key0.row_groups is not None,
        ROW_GROUPS1=key1.row_groups is not None,
        SPLIT_KV=split_kv,
        SPLIT_N=int(split_rows),
        BLOCK_N=64,
        BLOCK_D=128,
        num_warps=4,
        num_stages=2,
    )
    if split_kv:
        _merge_split_kv_attention_kernel[
            (batch * heads, triton.cdiv(channels, 16))
        ](
            partial_maximum,
            partial_denominator,
            partial_accumulator,
            output,
            output,
            output,
            num_splits=num_splits,
            channels=channels,
            APPLY_CORRECTION=False,
            BLOCK_S=merge_width,
            BLOCK_D=16,
            num_warps=4,
        )
    return output, logits if return_logits else None
