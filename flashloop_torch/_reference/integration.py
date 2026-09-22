#!/usr/bin/env python3
"""Compose decode cached-mass attention with token-level sparse prefill."""

from __future__ import annotations

from typing import Any

import torch

from .method_utils import method_uses_token_sparse_prefill
from .audit_utils import (
    validate_intervention_counts,
    validate_token_sparse_prefill_audit,
)


JointMassIntervention: Any | None = None
LoopFreezeIntervention: Any | None = None


def _resolve_interventions() -> tuple[Any, Any]:
    global JointMassIntervention, LoopFreezeIntervention
    if LoopFreezeIntervention is None:
        from .token_updates import LoopFreezeIntervention as resolved_prefill

        LoopFreezeIntervention = resolved_prefill
    if JointMassIntervention is None:
        from .attention import JointMassIntervention as resolved_attention

        JointMassIntervention = resolved_attention
    return LoopFreezeIntervention, JointMassIntervention


def token_sparse_prefill_metadata(
    *,
    method: str,
    attention_fraction: float,
    fraction_loop3: float,
    fraction_loop4: float,
) -> dict[str, Any]:
    """Return explicit row/environment fields for the joint method."""
    del attention_fraction
    enabled = method_uses_token_sparse_prefill(method)
    return {
        "token_sparse_prefill": enabled,
        "token_prefill_fraction_loop3": float(fraction_loop3) if enabled else None,
        "token_prefill_fraction_loop4": float(fraction_loop4) if enabled else None,
        "token_prefill_selector": "hidden_delta" if enabled else None,
    }


def validated_joint_audit(
    raw: dict[str, Any],
    *,
    generated_tokens: int,
    prompt_tokens: int,
    num_layers: int,
    fraction_loop3: float,
    fraction_loop4: float,
    recurrent_steps: int = 4,
) -> dict[str, Any]:
    """Validate and namespace the decode and prefill intervention receipts."""
    attention_raw = raw["attention"]
    attention_expected = validate_intervention_counts(
        generated_tokens=generated_tokens,
        num_layers=num_layers,
        source_captures=int(attention_raw["source_captures"]),
        target_replacements=int(attention_raw["target_replacements"]),
        recurrent_steps=recurrent_steps,
    )
    expected_prefill_calls = int(recurrent_steps) * int(num_layers)
    observed_prefill_calls = int(attention_raw["prefill_attention_calls"])
    if observed_prefill_calls != expected_prefill_calls:
        raise RuntimeError(
            "cached-mass prefill attention calls="
            f"{observed_prefill_calls}, expected={expected_prefill_calls}"
        )

    prefill_raw = raw["token_sparse_prefill"]
    prefill_expected = validate_token_sparse_prefill_audit(
        prompt_tokens=prompt_tokens,
        num_layers=num_layers,
        fraction_loop3=fraction_loop3,
        fraction_loop4=fraction_loop4,
        audit=prefill_raw,
        recurrent_steps=recurrent_steps,
    )
    return {
        "status": "PASS",
        "recurrent_steps": int(recurrent_steps),
        "prompt_tokens": int(prompt_tokens),
        "num_layers": int(num_layers),
        "attention": {
            **attention_expected,
            **attention_raw,
        },
        "token_sparse_prefill": {
            **prefill_expected,
            **prefill_raw,
        },
    }


def aggregate_validated_joint_audits(
    audits: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate request-local PASS receipts without weakening their contract."""
    if not audits:
        raise ValueError("joint audit aggregation requires at least one request")
    num_layers = int(audits[0].get("num_layers", -1))
    recurrent_steps = int(audits[0].get("recurrent_steps", 4))
    if num_layers <= 0:
        raise ValueError("joint audit is missing a positive layer count")
    for audit in audits:
        if audit.get("status") != "PASS":
            raise RuntimeError("joint audit aggregation received a non-PASS request")
        if int(audit.get("num_layers", -1)) != num_layers:
            raise RuntimeError("joint audit aggregation mixed layer counts")
        if int(audit.get("recurrent_steps", 4)) != recurrent_steps:
            raise RuntimeError("joint audit aggregation mixed recurrent depths")
        if audit.get("token_sparse_prefill", {}).get("mask_nested") is not True:
            raise RuntimeError("joint audit aggregation received a non-nested mask")

    observed = {
        "validated_requests": len(audits),
        "prompt_tokens": sum(int(audit["prompt_tokens"]) for audit in audits),
        "decode_steps": sum(
            int(audit["attention"]["decode_steps"]) for audit in audits
        ),
        "prefill_attention_calls": sum(
            int(audit["attention"]["prefill_attention_calls"]) for audit in audits
        ),
        "source_captures": sum(
            int(audit["attention"]["source_captures"]) for audit in audits
        ),
        "target_replacements": sum(
            int(audit["attention"]["target_replacements"]) for audit in audits
        ),
        "active_loop3_tokens": sum(
            int(audit["token_sparse_prefill"]["active_counts"]["3"])
            for audit in audits
        ),
        "active_loop4_tokens": sum(
            int(audit["token_sparse_prefill"]["active_counts"].get("4", 0))
            for audit in audits
        ),
        "cache_replacements": sum(
            int(audit["token_sparse_prefill"]["cache_replacements"])
            for audit in audits
        ),
        "frozen_layer_rows": sum(
            int(audit["token_sparse_prefill"]["frozen_layer_rows"])
            for audit in audits
        ),
        "dense_target_calls": sum(
            int(audit["token_sparse_prefill"]["dense_target_calls"])
            for audit in audits
        ),
    }
    denominator = recurrent_steps * observed["prompt_tokens"]
    if denominator <= 0:
        raise RuntimeError("joint audit aggregation observed no prompt tokens")
    active_tokens = 2 * observed["prompt_tokens"] + observed["active_loop3_tokens"]
    if recurrent_steps == 4:
        active_tokens += observed["active_loop4_tokens"]
    token_block_ratio = active_tokens / denominator
    return {
        "status": "PASS",
        "kind": "cached_mass_token_sparse_prefill",
        "recurrent_steps": recurrent_steps,
        "observed": observed,
        "expected": dict(observed),
        "all_masks_nested": True,
        "theoretical_token_block_ratio": token_block_ratio,
        "theoretical_token_block_reduction": 1.0 - token_block_ratio,
    }


class TokenSparsePrefillCachedMassIntervention:
    """Own both interventions and preserve their non-overlapping lifecycles."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        attention_fraction: float = 0.10,
        fraction_loop3: float = 0.25,
        fraction_loop4: float = 0.10,
    ) -> None:
        attention_fraction = float(attention_fraction)
        fraction_loop3 = float(fraction_loop3)
        fraction_loop4 = float(fraction_loop4)
        if not 0.0 < attention_fraction <= 1.0:
            raise ValueError("attention_fraction must be in (0, 1]")
        if not 0.0 < fraction_loop4 <= fraction_loop3 <= 1.0:
            raise ValueError("token prefill fractions require 0 < loop4 <= loop3 <= 1")

        self.attention_fraction = attention_fraction
        self.fraction_loop3 = fraction_loop3
        self.fraction_loop4 = fraction_loop4
        self.closed = False

        prefill_intervention, attention_intervention = _resolve_interventions()

        # Install the prefill forward wrappers first. The decode-only mass hook
        # then observes the final module output without replacing prefill calls.
        self.prefill = prefill_intervention(
            model,
            fraction_loop3=fraction_loop3,
            fraction_loop4=fraction_loop4,
            selector="hidden_delta",
        )
        try:
            self.attention = attention_intervention(
                model,
                method="cached_source_mass",
                fraction=attention_fraction,
                record_metrics=False,
            )
        except BaseException:
            self.prefill.close()
            self.closed = True
            raise

    def audit(self) -> dict[str, Any]:
        return {
            "attention": {
                "prefill_attention_calls": int(
                    self.attention.prefill_attention_calls
                ),
                "source_captures": int(self.attention.source_captures),
                "target_replacements": int(self.attention.target_replacements),
            },
            "token_sparse_prefill": self.prefill.audit(),
        }

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.attention.close()
        finally:
            try:
                self.prefill.close()
            finally:
                self.closed = True
