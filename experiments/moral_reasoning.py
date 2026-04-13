#!/usr/bin/env python3
"""
Moral-reasoning experiment.

For each (persona, scenario) pair, prompt an LLM to rate a moral dilemma
on a Likert scale and persist results to SQLite.

Usage
-----
    python -m experiments.moral_reasoning \
        --provider vllm --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --personas-file data/sampled_personas_2000.json \
        --scenarios-file data/claude-3-5-sonnet_AB_0_with_logprobs.jsonl \
        --use-db 1 --db-path results_moral.db

    python -m experiments.moral_reasoning \
        --provider openrouter \
        --api-key "$OPENROUTER_API_KEY" \
        --model claude-haiku-4.5 \
        --personas-file data/sampled_personas_2000.json \
        --scenarios-file data/claude-3-5-sonnet_AB_0_with_logprobs.jsonl \
        --use-db 1 --db-path results_moral.db
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from persona_sim.prompts import create_persona_prompt, create_scenario_prompt
from persona_sim.experiment import (
    extract_numeric_rating,
    query_llm,
    query_judge_llm,
    run_judge_phase,
    load_flagged_persona_ids,
    add_common_args,
    setup_clients,
    create_failure_tracker,
    record_failure,
    print_failure_summary,
)
from persona_sim.consts import RATING_SCALES

SYSTEM_PROMPT = (
    "We will do a role-playing game. You will be given a persona description "
    "and a scenario."
)


# ── Database ─────────────────────────────────────────────────────────────────

def init_results_db(db_path: str) -> sqlite3.Connection:
    """Create or open the moral-reasoning results database."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_json TEXT NOT NULL,
                scenario TEXT NOT NULL,
                score REAL,
                prompt TEXT,
                raw_response TEXT
            )
            """
        )
    return conn


def save_result_to_db(conn: sqlite3.Connection, result: dict):
    """Persist one evaluation result."""
    with conn:
        conn.execute(
            "INSERT INTO results "
            "(profile_json, scenario, score, prompt, raw_response) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                json.dumps(result["profile"]),
                result["scenario"],
                result["score"],
                result["prompt"],
                result.get("raw_response"),
            ),
        )


def get_completed_tasks(
    conn: sqlite3.Connection | None,
    thinking_model: bool = False,
) -> set:
    """Return the set of ``(profile_json, scenario)`` pairs already stored."""
    if conn is None:
        return set()
    cursor = conn.cursor()
    if thinking_model:
        cursor.execute(
            "SELECT profile_json, scenario FROM results "
            "WHERE raw_response IS NOT NULL OR score IS NOT NULL"
        )
    else:
        cursor.execute("SELECT profile_json, scenario FROM results")
    return {(row[0], row[1]) for row in cursor.fetchall()}


def get_all_results_from_db(conn: sqlite3.Connection | None) -> list[dict]:
    """Fetch every row as a list of result dicts."""
    if conn is None:
        return []
    cursor = conn.cursor()
    cursor.execute(
        "SELECT profile_json, scenario, score, prompt FROM results"
    )
    out: list[dict] = []
    for profile_json, scenario, score, prompt in cursor.fetchall():
        try:
            profile = json.loads(profile_json)
        except (json.JSONDecodeError, TypeError):
            profile = profile_json
        out.append({
            "profile": profile,
            "scenario": scenario,
            "score": score,
            "prompt": prompt,
        })
    return out


# ── Data loading ─────────────────────────────────────────────────────────────

def load_scenarios(path: str) -> list[str]:
    """Load unique moral-dilemma scenarios from a JSONL file."""
    scenarios: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                scenarios.append(json.loads(line)["scenario"])
    unique = list(set(scenarios))
    print(f"Loaded {len(unique)} unique scenarios from {path}")
    return unique


# ── Flagged-persona dummy injection ─────────────────────────────────────────

def inject_dummy_responses(
    conn: sqlite3.Connection,
    all_profiles: list[dict],
    scenarios: list[str],
    flagged_ids: set[int],
    completed_tasks: set,
) -> set:
    """
    Pre-fill DB rows for flagged personas so they are never sent to the API.

    Dummy rows use ``score = random(1, 5)`` seeded per-persona for
    reproducibility and ``raw_response = '__DUMMY_FLAGGED_PERSONA__'``.
    """
    if not flagged_ids or conn is None:
        return completed_tasks

    updated = set(completed_tasks)
    inserted = 0

    for profile in all_profiles:
        pid = profile.get("id")
        if pid is None or int(pid) not in flagged_ids:
            continue

        profile_json = json.dumps(profile)
        random.seed(int(pid))

        for scenario in scenarios:
            if (profile_json, scenario) in updated:
                continue
            with conn:
                conn.execute(
                    "INSERT INTO results "
                    "(profile_json, scenario, score, prompt, raw_response) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        profile_json,
                        scenario,
                        random.randint(1, 5),
                        None,
                        "__DUMMY_FLAGGED_PERSONA__",
                    ),
                )
            updated.add((profile_json, scenario))
            inserted += 1

    if inserted:
        print(
            f"[flagged-personas] Injected {inserted:,} dummy rows "
            f"(these will never be sent to the API)."
        )
    return updated


# ── Single-task evaluator ────────────────────────────────────────────────────

def _evaluate_single(
    profile: dict,
    scenario: str,
    client,
    model: str,
    provider: str,
    max_tokens: int,
) -> dict:
    """Evaluate one (persona, scenario) pair and return a result dict."""
    persona_description = profile["description"]
    prompt = create_scenario_prompt(scenario, persona_description)
    try:
        score, raw_response = query_llm(
            prompt,
            client,
            model=model,
            provider=provider,
            max_tokens=max_tokens,
            system_prompt=SYSTEM_PROMPT,
        )
        return {
            "profile": profile,
            "scenario": scenario,
            "score": score,
            "prompt": prompt,
            "raw_response": raw_response,
            "error": None,
        }
    except Exception as e:
        return {
            "profile": profile,
            "scenario": scenario,
            "score": None,
            "prompt": prompt,
            "raw_response": None,
            "error": str(e),
        }


# ── Output helpers ───────────────────────────────────────────────────────────

def save_results(results: list[dict], output_file: str = "results.json"):
    """Write the full result list to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the moral-reasoning persona experiment",
    )
    add_common_args(parser)
    parser.add_argument(
        "--use-db",
        type=int,
        default=0,
        help="Use SQLite database for persistence (0=no, 1=yes)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results.json",
        help="Path to save JSON results",
    )
    parser.add_argument(
        "--scenarios-file",
        type=str,
        default="data/claude-3-5-sonnet_AB_0_with_logprobs.jsonl",
        help="JSONL file containing moral-dilemma scenarios",
    )
    args = parser.parse_args()

    # ── Clients ──────────────────────────────────────────────────────────
    client, judge_client = setup_clients(args)

    # ── Load data ────────────────────────────────────────────────────────
    with open(args.personas_file, "r", encoding="utf-8") as f:
        all_profiles = json.load(f)
    if args.limit is not None:
        random.seed(42)
        random.shuffle(all_profiles)
        all_profiles = all_profiles[: args.limit]
    print(f"Loaded {len(all_profiles)} persona profiles from {args.personas_file}")

    scenarios = load_scenarios(args.scenarios_file)
    flagged_ids = load_flagged_persona_ids(args.flagged_personas_file)

    max_tokens = (
        args.thinking_max_tokens if args.thinking_model else args.max_tokens
    )
    conn = init_results_db(args.db_path) if args.use_db else None

    # ── Judge-only mode ──────────────────────────────────────────────────
    if args.judge_only:
        if conn is None:
            raise SystemExit("ERROR: --judge-only requires --use-db 1.")
        if judge_client is None or not args.judge_model:
            raise SystemExit(
                "ERROR: --judge-only requires --judge-model and a judge client."
            )
        run_judge_phase(
            conn, judge_client, args.judge_model,
            args.judge_provider, args.max_workers,
        )
        save_results(get_all_results_from_db(conn), args.output_file)
        conn.close()
        return

    # ── Phase 1: data collection ─────────────────────────────────────────
    completed_tasks = get_completed_tasks(
        conn, thinking_model=args.thinking_model,
    )

    if flagged_ids and conn is not None:
        completed_tasks = inject_dummy_responses(
            conn, all_profiles, scenarios, flagged_ids, completed_tasks,
        )

    if completed_tasks:
        print(f"Already completed: {len(completed_tasks):,} tasks")

    tasks: list[tuple[dict, str]] = []
    for profile in all_profiles:
        profile_json = json.dumps(profile)
        for scenario in scenarios:
            if (profile_json, scenario) in completed_tasks:
                continue
            tasks.append((profile, scenario))

    skipped = len(completed_tasks) if completed_tasks else 0
    print(
        f"Total tasks: {len(tasks) + skipped:,}  |  "
        f"Skipping: {skipped:,}  |  Running: {len(tasks):,}"
    )

    if not tasks:
        if conn is not None:
            save_results(get_all_results_from_db(conn), args.output_file)
            conn.close()
        return

    # Dry-run: verify the prompt before spending budget
    dry = _evaluate_single(
        tasks[0][0], tasks[0][1],
        client, args.model, args.provider, max_tokens,
    )
    print(f"\n--- Dry-run prompt ---\n{dry['prompt']}\n")

    tracker, failure_samples = create_failure_tracker()
    results: list[dict] = []
    saved_count = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {
            executor.submit(
                _evaluate_single,
                p, s, client, args.model, args.provider, max_tokens,
            ): (p, s)
            for p, s in tasks
        }

        with tqdm(
            as_completed(future_to_task),
            total=len(future_to_task),
            desc="Querying",
            unit="task",
        ) as pbar:
            for future in pbar:
                result = future.result()
                results.append(result)

                fail_type = None
                if result.get("error"):
                    fail_type = "api_error"
                elif args.thinking_model:
                    if result.get("raw_response") is None:
                        fail_type = "failed_extraction"
                elif result.get("score") is None:
                    fail_type = "failed_extraction"

                record_failure(tracker, failure_samples, fail_type, result)

                should_save = (
                    (args.thinking_model
                     and result.get("raw_response") is not None)
                    or (not args.thinking_model
                        and result.get("score") is not None)
                )
                if should_save and conn is not None:
                    save_result_to_db(conn, result)
                    saved_count += 1

                fail_total = (
                    tracker["failed_extraction"] + tracker["api_error"]
                )
                total_seen = tracker["success"] + fail_total
                pbar.set_postfix(
                    saved=saved_count,
                    failed=fail_total,
                    fail_rate=(
                        f"{100 * fail_total / total_seen:.0f}%"
                        if total_seen
                        else "0%"
                    ),
                    refresh=False,
                )

    print_failure_summary(tracker, failure_samples, len(results))

    # ── Phase 2: judge extraction (thinking models only) ─────────────────
    if args.thinking_model and conn is not None:
        if judge_client and args.judge_model:
            print("\nRunning judge phase to extract scores...")
            run_judge_phase(
                conn, judge_client, args.judge_model,
                args.judge_provider, args.max_workers,
            )
        else:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM results "
                "WHERE score IS NULL AND raw_response IS NOT NULL"
            )
            pending = cursor.fetchone()[0]
            if pending:
                print(
                    f"\nNote: {pending} raw responses stored but no judge "
                    f"configured. Re-run with --judge-only --judge-model "
                    f"<model> to extract scores."
                )

    # ── Save final results ───────────────────────────────────────────────
    if conn is not None:
        final_results = get_all_results_from_db(conn)
        conn.close()
    else:
        final_results = results

    save_results(final_results, args.output_file)


if __name__ == "__main__":
    main()
