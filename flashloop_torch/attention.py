#!/usr/bin/env python3
"""Compare official, cached-mass, and oracle-mass deltas on 10 Ouro prompts."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from .attention_utils import (
    DEFAULT_FRACTIONS,
    DenseAttentionCapture,
    atomic_delta_scores,
    attention_state,
    generate_tokens,
    make_decode_topk_mask,
    make_topk_mask,
    mass_scaled_sparse_output,
    reconstruction_metrics,
    scalar,
    selected_attention_mass,
    token_comparison,
    write_csv,
)


PROMPTS = (
    {
        "id": "discount_vat",
        "category": "arithmetic",
        "text": "Question: A shop discounts a 120 euro jacket by 25%, then adds 20% VAT to the discounted price. What is the final price? Explain briefly.\nAnswer:",
    },
    {
        "id": "train_speed",
        "category": "arithmetic",
        "text": "Question: A train travels 150 kilometers in 2.5 hours at constant speed. How far will it travel in 4 hours? Show the calculation.\nAnswer:",
    },
    {
        "id": "fraction_students",
        "category": "arithmetic",
        "text": "Question: Three fifths of a class of 35 students submitted an assignment early. How many students submitted early, and how many did not?\nAnswer:",
    },
    {
        "id": "number_pattern",
        "category": "pattern",
        "text": "Question: Continue the sequence 2, 6, 12, 20, 30 and explain the rule used to obtain the next two numbers.\nAnswer:",
    },
    {
        "id": "logic_mammals",
        "category": "logic",
        "text": "Question: All whales are mammals, and no mammals are insects. Can any whale be an insect? Explain the conclusion in one or two sentences.\nAnswer:",
    },
    {
        "id": "photosynthesis",
        "category": "science",
        "text": "Question: Why do plants generally need light for photosynthesis? Give a concise explanation suitable for a middle-school student.\nAnswer:",
    },
    {
        "id": "binary_search",
        "category": "coding",
        "text": "Question: In Python, what condition must a list satisfy before binary search can be applied correctly, and why? Answer briefly.\nAnswer:",
    },
    {
        "id": "compare_decimals",
        "category": "comparison",
        "text": "Question: Arrange 0.7, 0.07, 0.707, and 0.77 from smallest to largest, then state which is closest to 0.71.\nAnswer:",
    },
    {
        "id": "meeting_plan",
        "category": "planning",
        "text": "Question: A 30-minute project meeting must cover status, one blocking issue, and next steps. Propose a simple time allocation that totals 30 minutes.\nAnswer:",
    },
    {
        "id": "probability_balls",
        "category": "probability",
        "text": "Question: A bag contains 4 red, 3 blue, and 3 green balls. If one ball is drawn uniformly at random, what is the probability it is not blue?\nAnswer:",
    },
)

PAIR_SPECS = (
    (1, 2, "isolated_2_to_3"),
    (2, 3, "isolated_3_to_4"),
    (1, 3, "stale_loop2_cache_to_loop4"),
)
METHODS = ("official", "cached_source_mass", "oracle_target_mass")


def scales_for_method(
    method: str,
    source_mass: torch.Tensor,
    target_mass: torch.Tensor,
) -> tuple[torch.Tensor | float, torch.Tensor | float]:
    if method == "official":
        return 1.0, 1.0
    if method == "cached_source_mass":
        return source_mass, source_mass
    if method == "oracle_target_mass":
        return source_mass, target_mass
    raise ValueError(method)


def tensor_mean(value: torch.Tensor) -> float:
    return scalar(value.float().mean())


def analyze_pair(
    prompt: dict[str, str],
    decode_step: int,
    source_loop: int,
    target_loop: int,
    schedule: str,
    snapshot: list[list[dict[str, Any]]],
    model: torch.nn.Module,
    fractions: tuple[float, ...],
    reconstruction_rows: list[dict[str, Any]],
    mass_rows: list[dict[str, Any]],
) -> None:
    for layer_index, layer in enumerate(model.model.layers):
        source = snapshot[source_loop][layer_index]
        target = snapshot[target_loop][layer_index]
        allowed = source["allowed"]
        if not torch.equal(allowed, target["allowed"]):
            raise RuntimeError("source and target causal eligibility differ")
        allowed_full = allowed.view(1, 1, *allowed.shape).expand_as(source["logits"])
        oracle_scores = atomic_delta_scores(source, target).masked_fill(
            ~allowed_full, 0
        )
        oracle_energy = oracle_scores.sum().clamp_min(1e-30)
        for fraction in fractions:
            mask = make_topk_mask(source["weights"].float(), allowed, fraction)
            source_mass = selected_attention_mass(
                source["logits"], mask, allowed_full
            )
            target_mass = selected_attention_mass(
                target["logits"], mask, allowed_full
            )
            drift = target_mass - source_mass
            mass_rows.append(
                {
                    "prompt_id": prompt["id"],
                    "category": prompt["category"],
                    "decode_step": decode_step,
                    "schedule": schedule,
                    "layer": layer_index + 1,
                    "from_loop": source_loop + 1,
                    "to_loop": target_loop + 1,
                    "target_fraction": fraction,
                    "effective_fraction": scalar(
                        mask.sum() / allowed_full.sum()
                    ),
                    "source_mass_mean": tensor_mean(source_mass),
                    "target_mass_mean": tensor_mean(target_mass),
                    "mass_drift_mean": tensor_mean(drift),
                    "mass_abs_drift_mean": tensor_mean(drift.abs()),
                    "mass_relative_abs_drift_mean": tensor_mean(
                        drift.abs() / source_mass.clamp_min(1e-8)
                    ),
                    "source_mass_min_head": scalar(source_mass.min()),
                    "target_mass_min_head": scalar(target_mass.min()),
                }
            )
            for method in METHODS:
                source_scale, target_scale = scales_for_method(
                    method, source_mass, target_mass
                )
                approximation = mass_scaled_sparse_output(
                    source["output"],
                    source["logits"],
                    target["logits"],
                    source["values"],
                    target["values"],
                    mask,
                    source_scale,
                    target_scale,
                    layer.self_attn.o_proj.weight,
                )
                metrics = reconstruction_metrics(
                    source["output"],
                    target["output"],
                    approximation,
                    layer.input_layernorm_2,
                )
                row = {
                    "prompt_id": prompt["id"],
                    "category": prompt["category"],
                    "decode_step": decode_step,
                    "schedule": schedule,
                    "layer": layer_index + 1,
                    "from_loop": source_loop + 1,
                    "to_loop": target_loop + 1,
                    "method": method,
                    "target_fraction": fraction,
                    "effective_fraction": scalar(mask.sum() / allowed_full.sum()),
                    "source_mass_mean": tensor_mean(source_mass),
                    "target_mass_mean": tensor_mean(target_mass),
                    "mass_abs_drift_mean": tensor_mean(drift.abs()),
                    "oracle_atomic_energy_selected": scalar(
                        oracle_scores.masked_select(mask).sum() / oracle_energy
                    ),
                }
                row.update(metrics)
                reconstruction_rows.append(row)


def analyze_snapshot(
    prompt: dict[str, str],
    decode_step: int,
    snapshot: list[list[dict[str, Any]]],
    model: torch.nn.Module,
    fractions: tuple[float, ...],
    reconstruction_rows: list[dict[str, Any]],
    mass_rows: list[dict[str, Any]],
) -> None:
    for source_loop, target_loop, schedule in PAIR_SPECS:
        analyze_pair(
            prompt,
            decode_step,
            source_loop,
            target_loop,
            schedule,
            snapshot,
            model,
            fractions,
            reconstruction_rows,
            mass_rows,
        )


class JointMassIntervention:
    """Replace Loop-3/4 attention using legacy or adjacent-loop sparse caches."""

    def __init__(
        self,
        model: torch.nn.Module,
        method: str,
        fraction: float | None,
        transition_fractions: dict[str, float] | None = None,
        record_metrics: bool = True,
    ):
        if method not in METHODS:
            raise ValueError(method)
        if transition_fractions is None:
            if fraction is None or not 0.0 < float(fraction) <= 1.0:
                raise ValueError("fraction must be in (0, 1]")
            self.attention_transition_mode = "joint_loop2"
            self.fraction = float(fraction)
            self.transition_fractions = {
                "2->3": self.fraction,
                "2->4": self.fraction,
            }
        else:
            if fraction is not None:
                raise ValueError(
                    "set either fraction or transition_fractions, not both"
                )
            if set(transition_fractions) != {"2->3", "3->4"}:
                raise ValueError(
                    "transition_fractions must contain exactly 2->3 and 3->4"
                )
            self.transition_fractions = {
                transition: float(value)
                for transition, value in transition_fractions.items()
            }
            if any(
                not 0.0 < value <= 1.0
                for value in self.transition_fractions.values()
            ):
                raise ValueError("transition fractions must be in (0, 1]")
            self.attention_transition_mode = "adjacent"
            self.fraction = None
        self.method = method
        self.record_metrics = record_metrics
        self.num_layers = int(model.config.num_hidden_layers)
        self.source_states: dict[int, dict[str, Any]] = {}
        self.masks: dict[int, torch.Tensor] = {}
        self.source_masses: dict[int, torch.Tensor] = {}
        self.rows: list[dict[str, Any]] = []
        self.prefill_attention_calls = 0
        self.source_captures = 0
        self.target_replacements = 0
        self.handles = []
        for layer_index, layer in enumerate(model.model.layers):
            self.handles.append(
                layer.self_attn.register_forward_hook(
                    self._hook(layer_index), with_kwargs=True
                )
            )

    def _capture_source(
        self,
        layer_index: int,
        state: dict[str, Any],
        fraction: float,
        output_override: torch.Tensor | None = None,
    ) -> None:
        source = {
            key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in state.items()
        }
        if output_override is not None:
            source["output"] = output_override.detach()
        allowed = source["allowed"].view(
            1, 1, *source["allowed"].shape
        ).expand_as(source["logits"])
        mask = make_decode_topk_mask(source["weights"].float(), fraction)
        self.source_states[layer_index] = source
        self.masks[layer_index] = mask
        self.source_masses[layer_index] = selected_attention_mass(
            source["logits"], mask, allowed
        )
        self.source_captures += 1

    def _target_fraction(self, current_loop: int) -> float:
        if self.attention_transition_mode == "adjacent":
            return self.transition_fractions[f"{current_loop}->{current_loop + 1}"]
        return self.transition_fractions[f"2->{current_loop + 1}"]

    def _hook(self, layer_index: int):
        def hook(module, _inputs, kwargs, output):
            current_loop = int(kwargs["current_ut"])
            if output[1].shape[-2] != 1:
                self.prefill_attention_calls += 1
                return None
            if current_loop not in (1, 2, 3):
                return None
            state = attention_state(module, kwargs, output, self.num_layers)
            if current_loop == 1:
                self._capture_source(
                    layer_index,
                    state,
                    self.transition_fractions["2->3"],
                )
                return None

            source = self.source_states.get(layer_index)
            mask = self.masks.get(layer_index)
            source_mass = self.source_masses.get(layer_index)
            if source is None or mask is None or source_mass is None:
                raise RuntimeError(f"missing Loop-2 cache for layer {layer_index}")
            allowed = state["allowed"].view(
                1, 1, *state["allowed"].shape
            ).expand_as(state["logits"])
            target_mass = (
                selected_attention_mass(state["logits"], mask, allowed)
                if self.record_metrics or self.method == "oracle_target_mass"
                else source_mass
            )
            source_scale, target_scale = scales_for_method(
                self.method, source_mass, target_mass
            )
            approximation = mass_scaled_sparse_output(
                source["output"],
                source["logits"],
                state["logits"],
                source["values"],
                state["values"],
                mask,
                source_scale,
                target_scale,
                module.o_proj.weight,
            )
            dense_output = output[0]
            self.target_replacements += 1
            if self.attention_transition_mode == "adjacent" and current_loop == 2:
                self._capture_source(
                    layer_index,
                    state,
                    self.transition_fractions["3->4"],
                    output_override=approximation,
                )
            if not self.record_metrics:
                return approximation, output[1]
            error = dense_output.float() - approximation.float()
            dense_delta = dense_output.float() - source["output"].float()
            self.rows.append(
                {
                    "layer": layer_index + 1,
                    "from_loop": (
                        current_loop
                        if self.attention_transition_mode == "adjacent"
                        else 2
                    ),
                    "to_loop": current_loop + 1,
                    "method": self.method,
                    "fraction": self._target_fraction(current_loop),
                    "local_output_relative_l2_error": scalar(
                        torch.linalg.vector_norm(error)
                        / torch.linalg.vector_norm(dense_output.float()).clamp_min(1e-12)
                    ),
                    "local_delta_relative_l2_error": scalar(
                        torch.linalg.vector_norm(error)
                        / torch.linalg.vector_norm(dense_delta).clamp_min(1e-12)
                    ),
                    "source_mass_mean": tensor_mean(source_mass),
                    "target_mass_mean": tensor_mean(target_mass),
                    "mass_abs_drift_mean": tensor_mean(
                        (target_mass - source_mass).abs()
                    ),
                }
            )
            return approximation, output[1]

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.source_states.clear()
        self.masks.clear()
        self.source_masses.clear()


def mean(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def aggregate_results(
    reconstruction_rows: list[dict[str, Any]],
    mass_rows: list[dict[str, Any]],
    generation_rows: list[dict[str, Any]],
    fractions: tuple[float, ...],
    generation_fractions: tuple[float, ...],
) -> dict[str, Any]:
    offline: dict[str, Any] = {}
    for _, _, schedule in PAIR_SPECS:
        for method in METHODS:
            for fraction in fractions:
                selected = [
                    row
                    for row in reconstruction_rows
                    if row["schedule"] == schedule
                    and row["method"] == method
                    and row["target_fraction"] == fraction
                ]
                offline[f"{schedule}/{method}/f={fraction:.2f}"] = {
                    "row_count": len(selected),
                    "prompt_count": len({row["prompt_id"] for row in selected}),
                    "delta_energy_recovered_postnorm_mean": mean(
                        selected, "delta_energy_recovered_postnorm"
                    ),
                    "output_relative_l2_error_mean": mean(
                        selected, "output_relative_l2_error"
                    ),
                }
    mass: dict[str, Any] = {}
    for _, _, schedule in PAIR_SPECS:
        for fraction in fractions:
            selected = [
                row
                for row in mass_rows
                if row["schedule"] == schedule
                and row["target_fraction"] == fraction
            ]
            mass[f"{schedule}/f={fraction:.2f}"] = {
                "source_mass_mean": mean(selected, "source_mass_mean"),
                "target_mass_mean": mean(selected, "target_mass_mean"),
                "mass_abs_drift_mean": mean(selected, "mass_abs_drift_mean"),
                "mass_relative_abs_drift_mean": mean(
                    selected, "mass_relative_abs_drift_mean"
                ),
            }
    generation: dict[str, Any] = {}
    for method in METHODS:
        for fraction in generation_fractions:
            selected = [
                row
                for row in generation_rows
                if row["method"] == method and row["fraction"] == fraction
            ]
            generation[f"{method}/f={fraction:.2f}"] = {
                "prompt_count": len(selected),
                "exact_prompt_count": sum(
                    bool(row["exact_token_sequence_match"]) for row in selected
                ),
                "position_token_match_rate_mean": mean(
                    selected, "position_token_match_rate"
                ),
                "matching_prefix_tokens_mean": mean(
                    selected, "matching_prefix_tokens"
                ),
            }
    return {"offline": offline, "mass": mass, "generation": generation}


def run_self_test() -> None:
    assert len(PROMPTS) == 10
    assert len({prompt["id"] for prompt in PROMPTS}) == 10
    assert METHODS == ("official", "cached_source_mass", "oracle_target_mass")
    source_mass = torch.tensor([[[[0.6]]]])
    target_mass = torch.tensor([[[[0.7]]]])
    assert scales_for_method("cached_source_mass", source_mass, target_mass) == (
        source_mass,
        source_mass,
    )
    print("CHIPMUNK_MASS_SELF_TEST_OK")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--fractions", type=float, nargs="+", default=DEFAULT_FRACTIONS)
    parser.add_argument(
        "--generation-fractions", type=float, nargs="+", default=(0.10, 0.20, 0.30)
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if args.model is None or args.output_dir is None:
        parser.error("--model and --output-dir are required unless --self-test is used")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    fractions = tuple(float(value) for value in args.fractions)
    generation_fractions = tuple(float(value) for value in args.generation_fractions)
    if any(value <= 0 or value > 1 for value in (*fractions, *generation_fractions)):
        raise ValueError("fractions must be in (0, 1]")

    started = time.perf_counter()
    set_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).eval().to("cuda")
    num_loops = int(model.config.total_ut_steps)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    torch.cuda.reset_peak_memory_stats()

    reconstruction_rows: list[dict[str, Any]] = []
    mass_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    prompt_metadata: list[dict[str, Any]] = []

    with torch.inference_mode():
        for prompt_index, prompt in enumerate(PROMPTS, start=1):
            print(f"PROMPT_START {prompt_index}/10 {prompt['id']}")
            encoded = tokenizer(prompt["text"], return_tensors="pt")
            inputs = {key: value.to("cuda") for key, value in encoded.items()}
            capture = DenseAttentionCapture(model)
            prefill_output = model(
                **inputs, use_cache=True, logits_to_keep=1, return_dict=True
            )
            capture.take(num_loops)
            cache = prefill_output.past_key_values
            attention_mask = inputs["attention_mask"]
            next_token = prefill_output.logits[:, -1, :].argmax(
                dim=-1, keepdim=True
            )
            dense_generated: list[int] = []
            for decode_step in range(1, args.decode_tokens + 1):
                dense_generated.append(int(next_token.item()))
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], 1),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ],
                    dim=1,
                )
                decode_output = model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
                snapshot = capture.take(num_loops)
                analyze_snapshot(
                    prompt,
                    decode_step,
                    snapshot,
                    model,
                    fractions,
                    reconstruction_rows,
                    mass_rows,
                )
                cache = decode_output.past_key_values
                next_token = decode_output.logits[:, -1, :].argmax(
                    dim=-1, keepdim=True
                )
            capture.close()

            baseline_tokens = generate_tokens(
                model, inputs, args.decode_tokens, int(pad_token_id)
            )
            if baseline_tokens != dense_generated:
                raise RuntimeError(
                    f"manual/generate baseline mismatch for {prompt['id']}"
                )
            prompt_metadata.append(
                {
                    "id": prompt["id"],
                    "category": prompt["category"],
                    "prompt": prompt["text"],
                    "prompt_tokens": int(inputs["input_ids"].shape[-1]),
                    "dense_token_ids": baseline_tokens,
                    "dense_completion": tokenizer.decode(
                        baseline_tokens, skip_special_tokens=True
                    ),
                }
            )

            for method in METHODS:
                for fraction in generation_fractions:
                    intervention = JointMassIntervention(model, method, fraction)
                    candidate = generate_tokens(
                        model, inputs, args.decode_tokens, int(pad_token_id)
                    )
                    intervention.close()
                    comparison = token_comparison(baseline_tokens, candidate)
                    generation_rows.append(
                        {
                            "prompt_id": prompt["id"],
                            "category": prompt["category"],
                            "schedule": "joint_loop2_cache_to_loops3_and4",
                            "method": method,
                            "fraction": fraction,
                            "reference_tokens": len(baseline_tokens),
                            "generated_tokens": len(candidate),
                            **comparison,
                            "completion": tokenizer.decode(
                                candidate, skip_special_tokens=True
                            ),
                        }
                    )
                    for row in intervention.rows:
                        intervention_rows.append(
                            {
                                "prompt_id": prompt["id"],
                                "category": prompt["category"],
                                **row,
                            }
                        )
            print(f"PROMPT_DONE {prompt_index}/10 {prompt['id']}")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "mass_method_reconstruction.csv", reconstruction_rows)
    write_csv(args.output_dir / "selected_mass_drift.csv", mass_rows)
    write_csv(args.output_dir / "mass_generation_comparison.csv", generation_rows)
    write_csv(args.output_dir / "mass_intervention_local_metrics.csv", intervention_rows)
    aggregate = aggregate_results(
        reconstruction_rows,
        mass_rows,
        generation_rows,
        fractions,
        generation_fractions,
    )
    payload = {
        "metadata": {
            "model_path": str(args.model),
            "prompt_count": len(PROMPTS),
            "decode_tokens": args.decode_tokens,
            "fractions": list(fractions),
            "generation_fractions": list(generation_fractions),
            "methods": list(METHODS),
            "schedule": "Loop 2 dense refresh; same Loop-2 cache/mask independently serves Loops 3 and 4",
            "cached_mass": "source selected global probability mass scales both source and target conditional subset attention",
            "oracle_mass": "source and target exact selected global masses; non-deployable diagnostic, not guaranteed to upper-bound cached mass because approximation errors can cancel",
            "experiment_scope": "dense-instrumented PyTorch algorithm quality; no sparse kernel or speed claim",
            "git_commit": os.environ.get("OURO_GIT_COMMIT", "unknown"),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "dtype": str(next(model.parameters()).dtype),
            "elapsed_seconds": round(elapsed, 3),
            "peak_gpu_memory_gib": round(
                torch.cuda.max_memory_allocated() / (1024**3), 3
            ),
        },
        "prompts": prompt_metadata,
        **aggregate,
    }
    (args.output_dir / "chipmunk_mass_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(payload["metadata"], indent=2, ensure_ascii=False))
    print(json.dumps(aggregate["generation"], indent=2, ensure_ascii=False))
    print("OURO_CHIPMUNK_MASS_OK")
    return 0


