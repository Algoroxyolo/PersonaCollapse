#!/usr/bin/env python3
"""
Token and cost estimation for persona experiments.

Builds representative prompts from the actual data files, counts tokens,
and produces a cost report for every configured model/provider.

Usage
-----
    # Digital-twin (BFI) experiment
    python -m experiments.estimate_budget \\
        --experiment digital_twin \\
        --personas-file data/sampled_personas_2000.json --limit 2000

    # Moral-reasoning experiment
    python -m experiments.estimate_budget \\
        --experiment moral_reasoning \\
        --personas-file data/sampled_personas_2000.json

    # Filter to a single provider
    python -m experiments.estimate_budget \\
        --experiment digital_twin \\
        --personas-file data/sampled_personas_2000.json \\
        --provider openrouter
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any

from persona_sim.providers import (
    PROVIDER_REGISTRIES,
    resolve_model_id,
    OPENROUTER_BASE_URL,
    OPENAI_BASE_URL,
)
from persona_sim.prompts import (
    create_persona_prompt,
    create_bfi_prompt,
    create_scenario_prompt,
)

# ── System prompts (same as experiment code) ─────────────────────────────────

SYSTEM_DIGITAL_TWIN = "You are a helpful assistant acting out a specific persona."
SYSTEM_MORAL = (
    "We will do a role-playing game. You will be given a persona description "
    "and a scenario."
)


# ── Token counting ───────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """
    Estimate token count.

    Uses tiktoken (cl100k_base) when available; falls back to a ~3.5
    chars/token heuristic.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, int(len(text) / 3.5))


# ── Data loaders ─────────────────────────────────────────────────────────────

def load_personas(
    path: str,
    limit: int | None = None,
) -> list[dict]:
    """Load persona profiles from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    return data


def load_bfi_questions(
    path: str = "data/human_reference/BFI.json",
) -> dict[str, str]:
    """Load BFI questions ``{question_id: question_text}``."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["questions"]


def load_scenarios(path: str) -> list[str]:
    """Load unique scenarios from a JSONL file."""
    scenarios: list[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    scenarios.append(json.loads(line)["scenario"])
        return list(set(scenarios))
    except FileNotFoundError:
        return []


# ── Prompt sampling ──────────────────────────────────────────────────────────

def _sample_prompts_digital_twin(
    personas: list[dict],
    questions: dict[str, str],
    n_sample: int = 50,
) -> tuple[list[str], str]:
    """Build a sample of BFI prompts and return them with the system prompt."""
    prompts: list[str] = []
    q_items = list(questions.items())
    for p in personas[: max(1, n_sample // len(q_items) + 1)]:
        desc = create_persona_prompt(p)
        for _qid, qtxt in q_items:
            prompts.append(create_bfi_prompt(desc, qtxt))
            if len(prompts) >= n_sample:
                break
        if len(prompts) >= n_sample:
            break
    return prompts[:n_sample], SYSTEM_DIGITAL_TWIN


def _sample_prompts_moral(
    personas: list[dict],
    scenarios: list[str],
    n_sample: int = 50,
) -> tuple[list[str], str]:
    """Build a sample of moral-reasoning prompts."""
    prompts: list[str] = []
    for p in personas:
        desc = p["description"] if "description" in p else create_persona_prompt(p)
        for sc in scenarios:
            prompts.append(create_scenario_prompt(sc, desc))
            if len(prompts) >= n_sample:
                break
        if len(prompts) >= n_sample:
            break
    return prompts[:n_sample], SYSTEM_MORAL


# ── Core estimation ──────────────────────────────────────────────────────────

def compute_estimates(
    prompts: list[str],
    system_prompt: str,
    total_calls: int,
    max_output_tokens: int,
    provider: str | None = None,
) -> dict[str, Any]:
    """
    From a sample of prompts, estimate total tokens and costs per model.

    If *provider* is given, only that provider's models are included;
    otherwise all known providers are shown.
    """
    sys_tokens = estimate_tokens(system_prompt)
    sample_input_tokens = [estimate_tokens(p) + sys_tokens for p in prompts]
    avg_input = sum(sample_input_tokens) / len(sample_input_tokens)
    min_input = min(sample_input_tokens)
    max_input = max(sample_input_tokens)

    avg_output = max_output_tokens
    total_input = int(avg_input * total_calls)
    total_output = int(avg_output * total_calls)

    registries_to_show = (
        {provider: PROVIDER_REGISTRIES[provider]}
        if provider
        else PROVIDER_REGISTRIES
    )

    model_costs: dict[str, dict] = {}
    for prov_name, registry in registries_to_show.items():
        for name, (mid, ip, op) in registry.items():
            in_cost = (total_input / 1_000_000) * ip
            out_cost = (total_output / 1_000_000) * op
            model_costs[name] = {
                "id": mid,
                "provider": prov_name,
                "input_cost": round(in_cost, 2),
                "output_cost": round(out_cost, 2),
                "total_cost": round(in_cost + out_cost, 2),
            }

    return {
        "sample_size": len(prompts),
        "avg_input_tokens": round(avg_input, 1),
        "min_input_tokens": min_input,
        "max_input_tokens": max_input,
        "avg_output_tokens": avg_output,
        "total_calls": total_calls,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "model_costs": model_costs,
    }


# ── Report printer ───────────────────────────────────────────────────────────

_PROVIDER_ENV_KEYS: dict[str, tuple[str, str]] = {
    "bedrock":    ("AWS_BEARER_TOKEN_BEDROCK", "<your-bedrock-key>"),
    "openrouter": ("OPENROUTER_API_KEY",       "<your-openrouter-key>"),
    "openai":     ("OPENAI_API_KEY",           "<your-openai-key>"),
}


def format_report(
    experiment: str,
    est: dict,
    n_personas: int,
    n_items: int,
    item_label: str,
    personas_file: str,
    preferred_model: str | None = None,
    preferred_provider: str | None = None,
    limit: int | None = None,
    n_flagged: int = 0,
) -> str:
    """Build a human-readable budget estimation report."""
    lines: list[str] = []
    w = 72

    lines.append("=" * w)
    lines.append("  BUDGET ESTIMATION REPORT")
    lines.append("=" * w)
    lines.append(
        f"  Date           : {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    lines.append(f"  Experiment     : {experiment}")
    lines.append(f"  Personas file  : {personas_file}")
    if preferred_provider:
        lines.append(f"  Provider       : {preferred_provider}")
    if limit:
        lines.append(f"  Persona limit  : {limit}")
    lines.append("")

    lines.append("\u2500" * w)
    lines.append("  DATA SUMMARY")
    lines.append("\u2500" * w)
    lines.append(f"  Personas (total)      : {n_personas:,}")
    if n_flagged:
        lines.append(
            f"  Flagged (skipped)     : {n_flagged:,}  "
            f"\u2190 dummy responses, no API cost"
        )
        lines.append(
            f"  Personas (prompted)   : {n_personas - n_flagged:,}"
        )
    lines.append(f"  {item_label:22s}: {n_items:,}")
    lines.append(f"  Total API calls       : {est['total_calls']:,}")
    lines.append("")

    lines.append("\u2500" * w)
    lines.append(
        f"  TOKEN ESTIMATION  (sampled {est['sample_size']} prompts)"
    )
    lines.append("\u2500" * w)
    lines.append(
        f"  Avg input tokens/call : {est['avg_input_tokens']:,.0f}  "
        f"(min {est['min_input_tokens']:,}, "
        f"max {est['max_input_tokens']:,})"
    )
    lines.append(f"  Avg output tokens/call: {est['avg_output_tokens']}")
    lines.append(f"  Total input tokens    : {est['total_input_tokens']:,}")
    lines.append(f"  Total output tokens   : {est['total_output_tokens']:,}")
    lines.append("")

    lines.append("\u2500" * w)
    lines.append("  COST ESTIMATES BY MODEL")
    lines.append("\u2500" * w)
    header = (
        f"  {'Model':<22s} {'Provider':<12s} "
        f"{'Input $':>10s} {'Output $':>10s} {'Total $':>10s}"
    )
    lines.append(header)
    lines.append("  " + "\u2500" * (len(header) - 2))

    for name, info in est["model_costs"].items():
        prov = info.get("provider", "")
        marker = (
            " *" if preferred_model and name == preferred_model else ""
        )
        lines.append(
            f"  {name:<22s} {prov:<12s} "
            f"${info['input_cost']:>8,.2f}  "
            f"${info['output_cost']:>8,.2f}  "
            f"${info['total_cost']:>8,.2f}{marker}"
        )
    if preferred_model:
        lines.append("\n  * = selected model")
    lines.append("")

    model_arg = preferred_model or "claude-haiku-3.5"
    provider = preferred_provider or "openrouter"
    limit_arg = limit or n_personas
    env_var, env_placeholder = _PROVIDER_ENV_KEYS.get(
        provider, _PROVIDER_ENV_KEYS["openrouter"],
    )

    lines.append("\u2500" * w)
    lines.append("  EXAMPLE COMMANDS")
    lines.append("\u2500" * w)
    lines.append("  # Set your API key first:")
    lines.append(f'  export {env_var}="{env_placeholder}"')
    lines.append("")

    if experiment == "digital_twin":
        lines.append(
            f"  python -m experiments.digital_twin \\\n"
            f"      --provider {provider} \\\n"
            f"      --model {model_arg} \\\n"
            f"      --limit {limit_arg} \\\n"
            f'      --api-key "${env_var}" \\\n'
            f"      --personas-file {personas_file} \\\n"
            f"      --use-sampled-personas \\\n"
            f"      --db-path results_bfi_{model_arg}.db"
        )
    else:
        lines.append(
            f"  python -m experiments.moral_reasoning \\\n"
            f"      --provider {provider} \\\n"
            f"      --model {model_arg} \\\n"
            f'      --api-key "${env_var}" \\\n'
            f"      --personas-file {personas_file} \\\n"
            f"      --use-db 1 \\\n"
            f"      --db-path results_moral_{model_arg}.db"
        )
    lines.append("")
    lines.append("=" * w)
    return "\n".join(lines)


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate token usage and costs before running experiments."
        ),
    )
    parser.add_argument(
        "--experiment",
        required=True,
        choices=["digital_twin", "moral_reasoning"],
        help="Which experiment to estimate.",
    )
    parser.add_argument(
        "--personas-file",
        required=True,
        help="Path to personas JSON (e.g. sampled_personas_2000.json).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max personas to use (default: all).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Preferred model alias to highlight in the report.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=list(PROVIDER_REGISTRIES.keys()),
        help="Show costs only for this provider (default: all).",
    )
    parser.add_argument(
        "--scenarios-file",
        type=str,
        default="data/claude-3-5-sonnet_AB_0_with_logprobs.jsonl",
        help="Scenarios JSONL file (moral_reasoning only).",
    )
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=None,
        help="Override scenario count if file is unavailable.",
    )
    parser.add_argument(
        "--bfi-file",
        type=str,
        default="data/human_reference/BFI.json",
        help="BFI questions file (digital_twin only).",
    )
    parser.add_argument(
        "--save-report",
        type=str,
        default=None,
        help="Save the report to this file path.",
    )
    parser.add_argument(
        "--flagged-personas-file",
        type=str,
        default="data/flagged_personas.pkl",
        help=(
            "Pickle file of int persona IDs whose cost is excluded.  "
            "Pass '' to disable."
        ),
    )
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────
    if not os.path.exists(args.personas_file):
        print(f"Error: personas file not found: {args.personas_file}")
        sys.exit(1)

    personas = load_personas(args.personas_file, args.limit)
    n_personas = len(personas)
    print(f"Loaded {n_personas} personas from {args.personas_file}")

    # ── Flagged personas ─────────────────────────────────────────────────
    n_flagged = 0
    if args.flagged_personas_file and os.path.exists(
        args.flagged_personas_file,
    ):
        import pickle

        with open(args.flagged_personas_file, "rb") as fh:
            flagged_ids = {int(x) for x in pickle.load(fh)}
        persona_ids = {int(p["id"]) for p in personas if "id" in p}
        n_flagged = len(persona_ids & flagged_ids)
        print(
            f"[flagged-personas] {n_flagged:,} of {n_personas:,} personas "
            f"flagged (excluded from cost estimate)."
        )
    elif args.flagged_personas_file:
        print(
            f"[flagged-personas] Warning: {args.flagged_personas_file} "
            f"not found — cost estimate includes all personas."
        )

    n_personas_effective = n_personas - n_flagged

    # ── Experiment-specific setup ────────────────────────────────────────
    if args.experiment == "digital_twin":
        if not os.path.exists(args.bfi_file):
            print(f"Error: BFI file not found: {args.bfi_file}")
            sys.exit(1)
        questions = load_bfi_questions(args.bfi_file)
        n_items = len(questions)
        total_calls = n_personas_effective * n_items
        sample_prompts, sys_prompt = _sample_prompts_digital_twin(
            personas, questions,
        )
        item_label = "BFI questions"
        max_out = 100

    else:  # moral_reasoning
        scenarios = load_scenarios(args.scenarios_file)
        if not scenarios and args.n_scenarios:
            n_items = args.n_scenarios
            placeholder = (
                "A person faces a moral dilemma involving honesty vs. "
                "loyalty. They must decide whether to report a friend's "
                "wrongdoing. " * 3
            )
            scenarios = [placeholder] * min(n_items, 20)
            print(
                f"Scenarios file not found; using --n-scenarios={n_items} "
                f"with placeholder text for token estimation."
            )
        elif not scenarios:
            print(
                f"Error: scenarios file not found ({args.scenarios_file}) "
                f"and --n-scenarios not set."
            )
            sys.exit(1)
        else:
            n_items = len(scenarios)
            print(f"Loaded {n_items} scenarios")

        total_calls = n_personas_effective * n_items
        sample_prompts, sys_prompt = _sample_prompts_moral(
            personas, scenarios,
        )
        item_label = "Scenarios"
        max_out = 100

    # ── Compute ──────────────────────────────────────────────────────────
    est = compute_estimates(
        sample_prompts, sys_prompt, total_calls, max_out,
        provider=args.provider,
    )

    report = format_report(
        experiment=args.experiment,
        est=est,
        n_personas=n_personas,
        n_items=n_items,
        item_label=item_label,
        personas_file=args.personas_file,
        preferred_model=args.model,
        preferred_provider=args.provider,
        limit=args.limit,
        n_flagged=n_flagged,
    )

    print(report)

    if args.save_report:
        with open(args.save_report, "w") as f:
            f.write(report)
        print(f"Report saved to {args.save_report}")


if __name__ == "__main__":
    main()
