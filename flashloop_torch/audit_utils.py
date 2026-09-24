#!/usr/bin/env python3
"""Pure utilities for a resumable, public-protocol MATH-500 evaluation.

The prompt examples and answer normalization follow the public Minerva MATH
task in lm-evaluation-harness.  Keeping this module free of torch imports makes
the scoring and resume behavior testable on the login/local machine.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple


def generation_record(completion_ids: list[int]) -> dict[str, Any]:
    """Persist the exact generated token sequence for identity checks."""
    token_ids = [int(token_id) for token_id in completion_ids]
    return {
        "completion_token_ids": token_ids,
        "generated_tokens": len(token_ids),
    }


# Public four-shot Minerva prompt from EleutherAI/lm-evaluation-harness.
# The Ouro paper's separate in-house five-shot prompt has not been released.
MINERVA_FEWSHOT = (
    {
        "problem": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
        "solution": (
            "The expressions inside each square root must be non-negative. Therefore, "
            "$x-2 \\ge 0$, so $x\\ge2$, and $5-x \\ge 0$, so $x \\le 5$. Also, "
            "the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. "
            "Therefore, the domain of the expression is $\\boxed{[2,5)}$.\n"
            "Final Answer: The final answer is $[2,5)$. I hope it is correct."
        ),
    },
    {
        "problem": (
            "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find "
            "$\\det (\\mathbf{A} \\mathbf{B}).$"
        ),
        "solution": (
            "We have that $\\det (\\mathbf{A} \\mathbf{B}) = "
            "(\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\n"
            "Final Answer: The final answer is $24$. I hope it is correct."
        ),
    },
    {
        "problem": (
            "Terrell usually lifts two 20-pound weights 12 times. If he uses two "
            "15-pound weights instead, how many times must Terrell lift them in order "
            "to lift the same total weight?"
        ),
        "solution": (
            "If Terrell lifts two 20-pound weights 12 times, he lifts a total of "
            "$2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound "
            "weights instead for $n$ times, he will lift a total of "
            "$2\\cdot15\\cdot n=30n$ pounds of weight. Equating this to 480 pounds, "
            "we can solve for $n$:\n\\begin{align*}\n30n&=480\\\\\n"
            "\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\n"
            "Final Answer: The final answer is $16$. I hope it is correct."
        ),
    },
    {
        "problem": (
            "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\\\n"
            "6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ "
            "are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero."
        ),
        "solution": (
            "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n"
            "$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n"
            "$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\n"
            "Final Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct."
        ),
    },
)


def build_minerva_prompt(problem: str) -> str:
    """Render the fixed public Minerva few-shot completion prompt."""
    sections = []
    for example in MINERVA_FEWSHOT:
        sections.append(
            f"Problem:\n{example['problem']}\n\nSolution:{example['solution']}"
        )
    sections.append(f"Problem:\n{problem}\n\nSolution:")
    return "\n\n".join(sections)


def _last_boxed(text: str) -> Optional[str]:
    starts = [match.start() for match in re.finditer(r"\\(?:boxed|fbox)", text)]
    for start in reversed(starts):
        brace = text.find("{", start)
        if brace < 0:
            tail = text[start:].split("$", 1)[0]
            pieces = tail.split(maxsplit=1)
            if len(pieces) == 2:
                return pieces[1].strip()
            continue
        depth = 0
        for index in range(brace, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    return text[brace + 1 : index].strip()
    return None


def extract_final_answer(text: str) -> Optional[str]:
    """Extract Minerva's final-answer phrase, then fall back to the last box."""
    matches = list(
        re.finditer(
            r"Final Answer:\s*The final answer is\s*(.*?)\.\s*"
            r"I hope it is correct\.",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
    )
    if matches:
        return matches[-1].group(1).strip()
    return _last_boxed(text)


def _fix_fracs(value: str) -> str:
    parts = value.split("\\frac")
    rebuilt = parts[0]
    for part in parts[1:]:
        rebuilt += "\\frac"
        if not part or part[0] == "{":
            rebuilt += part
            continue
        if len(part) < 2:
            return value
        first, second, rest = part[0], part[1], part[2:]
        if second == "{":
            rebuilt += "{" + first + "}" + second + rest
        else:
            rebuilt += "{" + first + "}{" + second + "}" + rest
    return rebuilt


def _fix_sqrt(value: str) -> str:
    parts = value.split("\\sqrt")
    rebuilt = parts[0]
    for part in parts[1:]:
        if part and part[0] != "{":
            rebuilt += "\\sqrt{" + part[0] + "}" + part[1:]
        else:
            rebuilt += "\\sqrt" + part
    return rebuilt


def _fix_simple_slash(value: str) -> str:
    parts = value.split("/")
    if len(parts) != 2:
        return value
    try:
        numerator, denominator = int(parts[0]), int(parts[1])
    except ValueError:
        return value
    return f"\\frac{{{numerator}}}{{{denominator}}}"


def normalize_math_answer(value: Optional[str]) -> Optional[str]:
    """Apply formatting-only normalization used by strict MATH matching."""
    if value is None:
        return None
    value = value.replace("\n", "")
    value = value.replace("\\!", "").replace("\\\\", "\\")
    value = value.replace("tfrac", "frac").replace("dfrac", "frac")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("^{\\circ}", "").replace("^\\circ", "")
    value = value.replace("\\$", "").replace("$", "")
    if "\\text{ " in value:
        value = value.split("\\text{ ", 1)[0]
    value = value.replace("\\%", "").replace("%", "")
    value = value.replace(" .", " 0.").replace("{.", "{0.")
    if not value:
        return value
    if value[0] == ".":
        value = "0" + value
    if value.count("=") == 1 and len(value.split("=", 1)[0]) <= 2:
        value = value.split("=", 1)[1]
    value = _fix_sqrt(value)
    value = value.replace(" ", "")
    value = _fix_fracs(value)
    if value == "0.5":
        value = "\\frac{1}{2}"
    return _fix_simple_slash(value)


def normalized_strict_match(prediction: Optional[str], gold: Optional[str]) -> bool:
    """Compare normalized strings without symbolic algebra or an LLM judge."""
    if prediction is None or gold is None:
        return False
    return normalize_math_answer(prediction) == normalize_math_answer(gold)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    """Durably append one completed sample so interrupted jobs can resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def load_completed_keys(
    path: Path,
    *,
    required_fields: Dict[str, Any] | None = None,
) -> Set[Tuple[str, int]]:
    """Read completed rows and optionally require an exact resume configuration."""
    if not path.exists():
        return set()
    lines = path.read_text(encoding="utf-8").splitlines()
    completed: Set[Tuple[str, int]] = set()
    for line_index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if line_index == len(lines) - 1:
                continue
            raise
        if required_fields is not None:
            for field, expected in required_fields.items():
                observed = row.get(field)
                if observed != expected:
                    raise RuntimeError(
                        f"resume config {field}={observed!r}, expected={expected!r}"
                    )
        completed.add((str(row["method"]), int(row["dataset_index"])))
    return completed


def validate_intervention_counts(
    generated_tokens: int,
    num_layers: int,
    source_captures: int,
    target_replacements: int,
    attention_transition_mode: str = "joint_loop2",
    recurrent_steps: int = 4,
) -> Dict[str, int]:
    """Fail closed if the cached-mass hook did not cover every decode call."""
    decode_steps = max(0, int(generated_tokens) - 1)
    if attention_transition_mode not in ("joint_loop2", "adjacent"):
        raise ValueError(
            f"unsupported attention transition mode: {attention_transition_mode}"
        )
    recurrent_steps = int(recurrent_steps)
    if recurrent_steps < 3:
        raise ValueError("cached-mass intervention requires at least three loops")
    target_transitions = recurrent_steps - 2
    source_transitions = (
        target_transitions if attention_transition_mode == "adjacent" else 1
    )
    expected_source = decode_steps * int(num_layers) * source_transitions
    expected_target = decode_steps * int(num_layers) * target_transitions
    if int(source_captures) != expected_source:
        raise RuntimeError(
            f"source captures={source_captures}, expected={expected_source}"
        )
    if int(target_replacements) != expected_target:
        raise RuntimeError(
            f"target replacements={target_replacements}, expected={expected_target}"
        )
    return {
        "decode_steps": decode_steps,
        "expected_source_captures": expected_source,
        "expected_target_replacements": expected_target,
    }


def validate_sparse_mlp_counts(
    generated_tokens: int,
    num_layers: int,
    source_captures: int,
    target_replacements: int,
) -> Dict[str, int]:
    """Fail closed if sparse MLP did not cover decode Loops 2/3/4."""
    decode_steps = max(0, int(generated_tokens) - 1)
    expected_source = decode_steps * int(num_layers)
    expected_target = decode_steps * int(num_layers) * 2
    if int(source_captures) != expected_source:
        raise RuntimeError(
            "sparse MLP source captures="
            f"{source_captures}, expected={expected_source}"
        )
    if int(target_replacements) != expected_target:
        raise RuntimeError(
            "sparse MLP target replacements="
            f"{target_replacements}, expected={expected_target}"
        )
    return {
        "decode_steps": decode_steps,
        "expected_source_captures": expected_source,
        "expected_target_replacements": expected_target,
    }


def validate_token_sparse_prefill_audit(
    *,
    prompt_tokens: int,
    num_layers: int,
    fraction_loop3: float,
    fraction_loop4: float,
    audit: Dict[str, Any],
    recurrent_steps: int = 4,
) -> Dict[str, Any]:
    """Fail closed unless token-sparse prefill covered every active target loop."""
    prompt_tokens = int(prompt_tokens)
    num_layers = int(num_layers)
    fraction_loop3 = float(fraction_loop3)
    fraction_loop4 = float(fraction_loop4)
    recurrent_steps = int(recurrent_steps)
    if prompt_tokens <= 0 or num_layers <= 0:
        raise ValueError("prompt_tokens and num_layers must be positive")
    if not 0.0 < fraction_loop4 <= fraction_loop3 <= 1.0:
        raise ValueError("token prefill fractions require 0 < loop4 <= loop3 <= 1")
    if recurrent_steps not in (3, 4):
        raise ValueError("token-sparse prefill requires three or four loops")

    expected_loop3 = max(1, math.ceil(prompt_tokens * fraction_loop3))
    expected_loop4 = max(1, math.ceil(prompt_tokens * fraction_loop4))
    active_counts = audit.get("active_counts", {})
    observed_loop3 = int(active_counts.get("3", -1))
    observed_loop4 = int(active_counts.get("4", -1)) if recurrent_steps == 4 else None
    if observed_loop3 != expected_loop3 or (
        recurrent_steps == 4 and observed_loop4 != expected_loop4
    ):
        raise RuntimeError(
            "token sparse prefill active counts="
            f"{observed_loop3}/{observed_loop4}, expected="
            f"{expected_loop3}/{expected_loop4}"
        )

    expected_target_calls = (recurrent_steps - 2) * num_layers
    expected_replacements = (recurrent_steps - 2) * num_layers
    expected_frozen_rows = num_layers * (prompt_tokens - expected_loop3)
    if recurrent_steps == 4:
        expected_frozen_rows += num_layers * (prompt_tokens - expected_loop4)
    if int(audit.get("dense_target_calls", -1)) != expected_target_calls:
        raise RuntimeError(
            "token sparse prefill target calls="
            f"{audit.get('dense_target_calls')}, expected={expected_target_calls}"
        )
    if int(audit.get("cache_replacements", -1)) != expected_replacements:
        raise RuntimeError(
            "token sparse prefill cache replacements="
            f"{audit.get('cache_replacements')}, expected={expected_replacements}"
        )
    if int(audit.get("frozen_layer_rows", -1)) != expected_frozen_rows:
        raise RuntimeError(
            "token sparse prefill frozen layer rows="
            f"{audit.get('frozen_layer_rows')}, expected={expected_frozen_rows}"
        )
    if audit.get("mask_nested") is not True:
        raise RuntimeError("token sparse prefill masks are not nested")
    expected = {
        "expected_active_loop3": expected_loop3,
        "expected_target_calls": expected_target_calls,
        "expected_cache_replacements": expected_replacements,
        "expected_frozen_layer_rows": expected_frozen_rows,
    }
    if recurrent_steps == 4:
        expected["expected_active_loop4"] = expected_loop4
    return expected


def paired_accuracy_summary(
    dense_correctness: list[bool], cached_correctness: list[bool]
) -> Dict[str, Any]:
    """Summarize paired accuracy and an exact two-sided McNemar test."""
    if not dense_correctness or len(dense_correctness) != len(cached_correctness):
        raise ValueError("paired correctness vectors must be non-empty and equal length")
    pairs = list(zip(dense_correctness, cached_correctness))
    both_correct = sum(dense and cached for dense, cached in pairs)
    dense_only = sum(dense and not cached for dense, cached in pairs)
    cached_only = sum(cached and not dense for dense, cached in pairs)
    both_wrong = sum(not dense and not cached for dense, cached in pairs)
    count = len(pairs)
    dense_correct = both_correct + dense_only
    cached_correct = both_correct + cached_only
    discordant = dense_only + cached_only
    if discordant == 0:
        mcnemar_p = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(dense_only, cached_only) + 1)
        ) / (2**discordant)
        mcnemar_p = min(1.0, 2.0 * tail)
    return {
        "count": count,
        "dense_correct": dense_correct,
        "cached_correct": cached_correct,
        "dense_accuracy_pct": 100.0 * dense_correct / count,
        "cached_accuracy_pct": 100.0 * cached_correct / count,
        "accuracy_delta_pp": 100.0 * (cached_correct - dense_correct) / count,
        "both_correct": both_correct,
        "dense_only_correct": dense_only,
        "cached_only_correct": cached_only,
        "both_wrong": both_wrong,
        "discordant": discordant,
        "mcnemar_exact_p_two_sided": mcnemar_p,
    }
