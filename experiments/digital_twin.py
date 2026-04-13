#!/usr/bin/env python3
"""
Digital-twin (BFI personality-test replication) experiment.

For each (persona, BFI question) pair, prompt an LLM to respond on a 1-5
agree/disagree scale and store results in SQLite for comparison with human
survey data.

Usage
-----
    # With sampled personas
    python -m experiments.digital_twin \
        --provider vllm --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --use-sampled-personas \
        --personas-file data/sampled_personas_2000.json \
        --db-path results_bfi.db

    # With human CSV data
    python -m experiments.digital_twin \
        --provider openrouter \
        --api-key "$OPENROUTER_API_KEY" \
        --model claude-haiku-4.5 \
        --limit 200 \
        --db-path results_bfi.db
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from persona_sim.prompts import create_persona_prompt, create_bfi_prompt
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

# ── Constants ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = "You are a helpful assistant acting out a specific persona."

BFI_JSON_PATH = os.path.join("data", "human_reference", "BFI.json")
CSV_PATH = os.path.join("data", "human_reference", "wave_1_numbers.csv")

BFI_SCALE = 5

COLUMN_MAP = {
    "Q11": "Region",
    "Q12": "Gender",
    "Q13": "Age",
    "Q14": "Education",
    "Q15": "Race",
    "Q16": "Citizenship",
    "Q17": "Self Description",
    "Q18": "Religion",
    "Q19": "Religious Attendance",
    "Q20": "Political Party",
    "Q21": "Income",
    "Q22": "Political Views",
    "Q23": "Household Size",
    "Q24": "Employment Status",
}


# ── Database ─────────────────────────────────────────────────────────────────

def init_db(db_path: str) -> sqlite3.Connection:
    """Create or open the BFI responses database."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bfi_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                twin_id TEXT,
                question_id TEXT,
                score INTEGER,
                profile_json TEXT,
                raw_response TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_twin_q "
            "ON bfi_responses(twin_id, question_id)"
        )
    return conn


def get_completed(
    conn: sqlite3.Connection,
    thinking_model: bool = False,
) -> set[tuple[str, str]]:
    """Return ``{(twin_id, question_id)}`` pairs already in the DB."""
    cursor = conn.cursor()
    if thinking_model:
        cursor.execute(
            "SELECT twin_id, question_id FROM bfi_responses "
            "WHERE raw_response IS NOT NULL OR score IS NOT NULL"
        )
    else:
        cursor.execute(
            "SELECT twin_id, question_id FROM bfi_responses"
        )
    return {(row[0], row[1]) for row in cursor.fetchall()}


def save_result(
    conn: sqlite3.Connection,
    result: dict,
    thinking_model: bool = False,
):
    """
    Persist one BFI result.

    Standard mode: only rows with a valid score are saved.
    Thinking-model mode: rows with a raw_response are saved (score may
    be filled later by the judge phase).
    """
    if thinking_model:
        if result.get("raw_response") is None:
            return
    else:
        if result["score"] is None:
            return

    with conn:
        conn.execute(
            "INSERT INTO bfi_responses "
            "(twin_id, question_id, score, profile_json, raw_response) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                result["profile"]["TWIN_ID"],
                result["question_id"],
                result["score"],
                json.dumps(result["profile"]),
                result.get("raw_response"),
            ),
        )


# ── Data loading ─────────────────────────────────────────────────────────────

def load_bfi_questions(
    path: str = BFI_JSON_PATH,
) -> dict[str, str]:
    """Load BFI question mapping ``{question_id: question_text}``."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def load_human_data(limit: int | None = None) -> list[dict]:
    """Load persona profiles from the original human-survey CSV."""
    print("Loading human data from CSV...")
    df = pd.read_csv(CSV_PATH, header=0, skiprows=[1])

    profiles: list[dict] = []
    for _, row in df.iterrows():
        profile: dict = {"TWIN_ID": row["TWIN_ID"]}
        for col_code, col_name in COLUMN_MAP.items():
            if col_code in row and pd.notna(row[col_code]):
                profile[col_name] = str(row[col_code])
        profiles.append(profile)
        if limit and len(profiles) >= limit:
            break

    print(f"Loaded {len(profiles)} profiles from CSV.")
    return profiles


def load_sampled_personas(
    file_path: str,
    limit: int | None = None,
) -> list[dict]:
    """Load persona profiles from a JSON file."""
    print(f"Loading sampled personas from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if limit is not None:
        random.seed(42)
        random.shuffle(data)

    profiles: list[dict] = []
    for item in data:
        if "id" in item and "TWIN_ID" not in item:
            item["TWIN_ID"] = str(item["id"])
        profiles.append(item)
        if limit and len(profiles) >= limit:
            break

    print(f"Loaded {len(profiles)} sampled personas.")
    return profiles


# ── Flagged-persona dummy injection ─────────────────────────────────────────

def inject_dummy_bfi_responses(
    conn: sqlite3.Connection,
    profiles: list[dict],
    questions: dict[str, str],
    flagged_ids: set[int],
    completed: set[tuple[str, str]],
) -> set[tuple[str, str]]:
    """
    Pre-fill DB rows for flagged personas so they are never queried.

    Dummy rows use ``score = random(1, 5)`` seeded per-persona and
    ``raw_response = '__DUMMY_FLAGGED_PERSONA__'``.
    """
    if not flagged_ids or conn is None:
        return completed

    updated = set(completed)
    inserted = 0

    for profile in profiles:
        pid = profile.get("id")
        if pid is None or int(pid) not in flagged_ids:
            continue

        tid = str(profile["TWIN_ID"])
        random.seed(int(pid))

        for q_id in questions:
            if (tid, str(q_id)) in updated:
                continue
            with conn:
                conn.execute(
                    "INSERT INTO bfi_responses "
                    "(twin_id, question_id, score, profile_json, raw_response) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        tid,
                        str(q_id),
                        random.randint(1, 5),
                        json.dumps(profile),
                        "__DUMMY_FLAGGED_PERSONA__",
                    ),
                )
            updated.add((tid, str(q_id)))
            inserted += 1

    if inserted:
        print(
            f"[flagged-personas] Injected {inserted:,} dummy BFI rows "
            f"(these will never be sent to the API)."
        )
    return updated


# ── Single-task evaluator ────────────────────────────────────────────────────

def _evaluate_single(
    profile: dict,
    question_id: str,
    question_text: str,
    client,
    model: str,
    provider: str,
    max_tokens: int,
) -> dict:
    """Evaluate one (persona, BFI question) pair."""
    persona_desc = create_persona_prompt(profile)
    prompt = create_bfi_prompt(persona_desc, question_text)
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
            "question_id": question_id,
            "question_text": question_text,
            "score": score,
            "raw_response": raw_response,
            "error": None,
        }
    except Exception as e:
        return {
            "profile": profile,
            "question_id": question_id,
            "question_text": question_text,
            "score": None,
            "raw_response": None,
            "error": str(e),
        }


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the BFI digital-twin personality-test experiment",
    )
    add_common_args(parser)
    parser.add_argument(
        "--use-sampled-personas",
        action="store_true",
        help="Use sampled personas JSON instead of human CSV data",
    )
    parser.add_argument(
        "--personas-file",
        type=str,
        default="data/sampled_personas_2000.json",
        help="Path to sampled personas JSON file",
    )
    args = parser.parse_args()

    # ── Clients ──────────────────────────────────────────────────────────
    client, judge_client = setup_clients(args)

    # ── Load profiles ────────────────────────────────────────────────────
    if args.use_sampled_personas:
        profiles = load_sampled_personas(args.personas_file, args.limit)
    else:
        profiles = load_human_data(args.limit)

    questions = load_bfi_questions()
    flagged_ids = load_flagged_persona_ids(args.flagged_personas_file)

    max_tokens = (
        args.thinking_max_tokens if args.thinking_model else args.max_tokens
    )

    conn = init_db(args.db_path)

    # ── Judge-only mode ──────────────────────────────────────────────────
    if args.judge_only:
        if judge_client is None or not args.judge_model:
            raise SystemExit(
                "ERROR: --judge-only requires --judge-model and a judge client."
            )
        print("Judge-only mode: extracting scores from stored raw responses...")
        run_judge_phase(
            conn, judge_client, args.judge_model,
            args.judge_provider, args.max_workers,
            table="bfi_responses",
        )
        conn.close()
        return

    # ── Phase 1: data collection ─────────────────────────────────────────
    completed = get_completed(conn, thinking_model=args.thinking_model)

    if flagged_ids:
        completed = inject_dummy_bfi_responses(
            conn, profiles, questions, flagged_ids, completed,
        )

    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM bfi_responses WHERE score IS NOT NULL"
    )
    valid_in_db = cur.fetchone()[0]
    print(
        f"Tasks in DB (completed set): {len(completed)}  |  "
        f"rows with valid scores: {valid_in_db}"
    )
    print(f"Profiles: {len(profiles)}  |  Questions: {len(questions)}")

    tasks: list[tuple[dict, str, str]] = []
    for profile in profiles:
        tid = profile["TWIN_ID"]
        for q_id, q_text in questions.items():
            if (str(tid), str(q_id)) in completed:
                continue
            tasks.append((profile, q_id, q_text))

    print(f"Remaining tasks: {len(tasks):,}")

    if not tasks:
        cur.execute(
            "SELECT COUNT(*) FROM bfi_responses WHERE score IS NOT NULL"
        )
        print(
            f"All tasks completed. Rows with valid scores: "
            f"{cur.fetchone()[0]}"
        )
    else:
        tracker, failure_samples = create_failure_tracker()
        saved_count = 0

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_task = {
                executor.submit(
                    _evaluate_single,
                    t[0], t[1], t[2],
                    client, args.model, args.provider, max_tokens,
                ): t
                for t in tasks
            }

            with tqdm(
                as_completed(future_to_task),
                total=len(future_to_task),
                desc="Querying",
                unit="task",
            ) as pbar:
                for future in pbar:
                    result = future.result()

                    fail_type = None
                    if result.get("error"):
                        fail_type = "api_error"
                    elif args.thinking_model:
                        if result.get("raw_response") is None:
                            fail_type = "failed_extraction"
                    elif result.get("score") is None:
                        fail_type = "failed_extraction"

                    record_failure(
                        tracker, failure_samples, fail_type, result,
                    )

                    if fail_type is None:
                        save_result(
                            conn, result,
                            thinking_model=args.thinking_model,
                        )
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

        print_failure_summary(tracker, failure_samples, len(tasks))

        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM bfi_responses WHERE score IS NOT NULL"
        )
        print(
            f"Rows with valid scores in DB (ground truth): "
            f"{cur.fetchone()[0]}"
        )

    # ── Phase 2: judge extraction (thinking models only) ─────────────────
    if args.thinking_model:
        if judge_client and args.judge_model:
            print(
                "\nRunning judge phase to extract scores from "
                "thinking model outputs..."
            )
            run_judge_phase(
                conn, judge_client, args.judge_model,
                args.judge_provider, args.max_workers,
                table="bfi_responses",
            )
        else:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM bfi_responses "
                "WHERE score IS NULL AND raw_response IS NOT NULL"
            )
            pending = cursor.fetchone()[0]
            if pending:
                print(
                    f"\nNote: {pending} raw responses stored but no judge "
                    f"configured. Re-run with --judge-only --judge-model "
                    f"<model> to extract scores."
                )

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
