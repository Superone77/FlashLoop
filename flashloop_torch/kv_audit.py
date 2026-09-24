#!/usr/bin/env python3
"""Torch-free validation for cross-loop KV accounting receipts."""

from __future__ import annotations

from typing import Any


def validated_cross_loop_kv_audit(
    raw: dict[str, Any],
    *,
    generated_tokens: int,
    prompt_tokens: int,
    num_layers: int,
    recurrent_steps: int = 4,
) -> dict[str, Any]:
    recurrent_steps = int(recurrent_steps)
    if not 1 <= recurrent_steps <= 4:
        raise ValueError("recurrent_steps must be in [1, 4]")
    observed_recurrent_steps = int(raw.get("recurrent_steps", 4))
    if observed_recurrent_steps != recurrent_steps:
        raise RuntimeError(
            f"recurrent_steps={observed_recurrent_steps}, expected={recurrent_steps}"
        )
    expected_forwards = int(generated_tokens)
    expected_cache_tokens = int(prompt_tokens) + max(0, int(generated_tokens) - 1)
    for field in ("forward_calls", "cache_updates"):
        observed = int(raw.get(field, -1))
        if observed != expected_forwards:
            raise RuntimeError(f"{field}={observed}, expected={expected_forwards}")
    if int(raw.get("total_cache_tokens", -1)) != expected_cache_tokens:
        raise RuntimeError(
            f"total_cache_tokens={raw.get('total_cache_tokens')}, "
            f"expected={expected_cache_tokens}"
        )
    group_size = int(raw.get("group_size", 0))
    quantized_tokens = int(raw.get("quantized_tokens", -1))
    if group_size <= 0 or quantized_tokens < 0 or quantized_tokens % group_size:
        raise RuntimeError("invalid group_size/quantized_tokens accounting")
    expected_groups = quantized_tokens // group_size * int(num_layers)
    if int(raw.get("quantized_groups", -1)) != expected_groups:
        raise RuntimeError(
            f"quantized_groups={raw.get('quantized_groups')}, expected={expected_groups}"
        )
    tail = expected_cache_tokens - quantized_tokens
    if int(raw.get("bf16_tail_tokens", -1)) != tail:
        raise RuntimeError(
            f"bf16_tail_tokens={raw.get('bf16_tail_tokens')}, expected={tail}"
        )
    packed = int(raw.get("total_packed_bytes", 0))
    dense = int(
        raw.get("dense_recurrent_bf16_bytes")
        or raw.get("dense_r4_bf16_bytes", 0)
    )
    if packed <= 0 or dense <= 0 or packed >= dense:
        raise RuntimeError("packed byte accounting does not represent compression")
    if raw.get("reference_cache_materialized") is not True:
        raise RuntimeError("fake-quant materialization boundary is missing")
    return {
        **raw,
        "status": "PASS",
        "recurrent_steps": recurrent_steps,
        "expected": {
            "forward_calls": expected_forwards,
            "cache_updates": expected_forwards,
            "total_cache_tokens": expected_cache_tokens,
            "quantized_groups": expected_groups,
            "bf16_tail_tokens": tail,
        },
    }


def aggregate_cross_loop_kv_audits(audits: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate independently validated request receipts without averaging ratios."""
    if not audits:
        raise ValueError("cross-loop KV aggregation requires at least one request")
    for audit in audits:
        if audit.get("status") != "PASS":
            raise RuntimeError("cross-loop KV aggregation received a non-PASS audit")
        if audit.get("reference_cache_materialized") is not True:
            raise RuntimeError("cross-loop KV audit lost its materialization boundary")
    profiles = {audit.get("codec_profile") for audit in audits}
    if len(profiles) != 1:
        raise RuntimeError("cross-loop KV aggregation mixed codec profiles")
    group_sizes = {int(audit.get("group_size", -1)) for audit in audits}
    residual_lengths = {int(audit.get("residual_length", -1)) for audit in audits}
    if len(group_sizes) != 1 or len(residual_lengths) != 1:
        raise RuntimeError("cross-loop KV aggregation mixed codec settings")
    recurrent_steps_set = {int(audit.get("recurrent_steps", 4)) for audit in audits}
    if len(recurrent_steps_set) != 1:
        raise RuntimeError("cross-loop KV aggregation mixed recurrent depths")
    recurrent_steps = next(iter(recurrent_steps_set))
    packed = sum(int(audit["total_packed_bytes"]) for audit in audits)
    dense = sum(
        int(
            audit.get("dense_recurrent_bf16_bytes")
            or audit.get("dense_r4_bf16_bytes", 0)
        )
        for audit in audits
    )
    if packed <= 0 or dense <= 0 or packed >= dense:
        raise RuntimeError("aggregated KV byte accounting is invalid")
    return {
        "status": "PASS",
        "kind": "cross_loop_kv_fake_quant",
        "codec_profile": next(iter(profiles)),
        "recurrent_steps": recurrent_steps,
        "group_size": next(iter(group_sizes)),
        "residual_length": next(iter(residual_lengths)),
        "requests": len(audits),
        "forward_calls": sum(int(audit["forward_calls"]) for audit in audits),
        "cache_updates": sum(int(audit["cache_updates"]) for audit in audits),
        "quantized_groups": sum(int(audit["quantized_groups"]) for audit in audits),
        "quantized_tokens_sum": sum(int(audit["quantized_tokens"]) for audit in audits),
        "total_cache_tokens_sum": sum(int(audit["total_cache_tokens"]) for audit in audits),
        "total_packed_bytes": packed,
        "dense_recurrent_bf16_bytes": dense,
        "dense_r4_bf16_bytes": dense if recurrent_steps == 4 else None,
        "dense_recurrent_ratio": packed / dense,
        "dense_r4_ratio": packed / dense if recurrent_steps == 4 else None,
        "theoretical_packed_byte_reduction": 1.0 - packed / dense,
        "reference_cache_materialized": True,
    }
