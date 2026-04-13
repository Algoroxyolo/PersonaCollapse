"""Shared experiment-runner infrastructure.

Provides LLM query helpers, judge extraction, failure tracking, and the
common argparse setup used by both the moral-reasoning and digital-twin
(BFI) experiments.
"""

import argparse
import os
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from persona_sim.providers import create_client

# ---------------------------------------------------------------------------
# Rating extraction
# ---------------------------------------------------------------------------


def extract_numeric_rating(text, max_scale=5):
    """Pull a 1-to-*max_scale* integer from *text*.

    Tries ``score: N`` / ``rating: N`` patterns first, then falls back to
    the first bare integer in range.
    """
    explicit_match = re.search(r'(score|rating)\D+(\d+)', text, flags=re.IGNORECASE)
    if explicit_match:
        value = int(explicit_match.group(2))
        if 1 <= value <= max_scale:
            return value

    numbers = re.findall(r'\d+', text)
    for num_str in numbers:
        value = int(num_str)
        if 1 <= value <= max_scale:
            return value

    return None


# ---------------------------------------------------------------------------
# LLM query helpers
# ---------------------------------------------------------------------------


def query_llm(prompt, client, model, system_prompt, provider="openai", max_tokens=512):
    """Send *prompt* to an LLM and attempt to extract a numeric rating.

    Retries up to 5 times with exponential back-off on transient errors.

    Returns:
        ``(score_or_none, raw_response_text, full_response_object)``
    """
    extra = (
        {"max_completion_tokens": max_tokens}
        if provider == "openai"
        else {"max_tokens": max_tokens, "temperature": 0.7}
    )

    max_retries = 5
    base_delay = 2

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                **extra,
            )
            rating_text = response.choices[0].message.content.strip()
            rating = extract_numeric_rating(rating_text)
            return rating, rating_text, response

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = base_delay * (2 ** attempt)
                time.sleep(wait_time)
            else:
                raise e

    return None, None, None


def query_judge_llm(raw_response, client, model, provider="openai", max_scale=5):
    """Ask a judge LLM to extract the final numeric rating from a verbose response.

    Returns an integer in ``[1, max_scale]`` or ``None``.
    """
    judge_prompt = (
        f"You are an extraction assistant. An AI was asked to rate something on a scale of "
        f"1 to {max_scale}. Below is its full response, which may contain extended reasoning "
        f"or a thinking chain before the final answer.\n"
        f"Your task: identify and output ONLY the final numeric rating (an integer from 1 to {max_scale}). "
        f"Respond with a single integer and nothing else.\n\n"
        f"Response to extract from:\n{raw_response}"
    )

    extra = (
        {"max_completion_tokens": 20}
        if provider == "openai"
        else {"max_tokens": 20, "temperature": 0.0}
    )
    max_retries = 5
    base_delay = 2

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": judge_prompt}],
                **extra,
            )
            text = response.choices[0].message.content.strip()
            return extract_numeric_rating(text, max_scale=max_scale)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
            else:
                print(f"Judge error after {max_retries} retries: {e}")
                return None
    return None


# ---------------------------------------------------------------------------
# Judge phase
# ---------------------------------------------------------------------------


def run_judge_phase(
    conn,
    table_name,
    judge_client,
    judge_model,
    judge_provider,
    max_workers,
    max_scale=5,
):
    """Run the judge extraction phase on rows with NULL scores.

    Queries all rows in *table_name* where ``score IS NULL`` but
    ``raw_response IS NOT NULL``, then updates each row with the
    extracted score.

    Returns:
        The number of scores successfully extracted.
    """
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT id, raw_response FROM {table_name} "
        "WHERE score IS NULL AND raw_response IS NOT NULL"
    )
    pending = cursor.fetchall()

    if not pending:
        print("Judge phase: no pending rows found.")
        return 0

    print(f"Judge phase: extracting scores for {len(pending)} rows...")

    def _judge_single(row_id, raw_text):
        score = query_judge_llm(
            raw_text, judge_client, judge_model, judge_provider, max_scale=max_scale
        )
        return row_id, score

    success = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_judge_single, row[0], row[1]): row[0]
            for row in pending
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Judge extraction"):
            row_id, score = future.result()
            if score is not None:
                with conn:
                    conn.execute(
                        f"UPDATE {table_name} SET score=? WHERE id=?",
                        (score, row_id),
                    )
                success += 1

    print(f"Judge phase complete: {success}/{len(pending)} scores extracted.")
    return success


# ---------------------------------------------------------------------------
# Flagged-persona loader
# ---------------------------------------------------------------------------


def load_flagged_persona_ids(flagged_file):
    """Load a set of integer persona IDs from a pickle file.

    Returns an empty set when the file is missing or the path is falsy.
    """
    if not flagged_file or not os.path.exists(flagged_file):
        if flagged_file:
            print(f"[flagged-personas] Warning: {flagged_file} not found — no personas skipped.")
        return set()

    with open(flagged_file, "rb") as fh:
        ids = pickle.load(fh)
    result = set(int(x) for x in ids)
    print(f"[flagged-personas] Loaded {len(result):,} flagged persona IDs from {flagged_file}")
    return result


# ---------------------------------------------------------------------------
# Failure tracking
# ---------------------------------------------------------------------------


def create_failure_tracker():
    """Return a fresh failure-tracking dict."""
    return {
        "stats": {"success": 0, "failed_extraction": 0, "api_error": 0},
        "samples": {"api_error": [], "failed_extraction": []},
        "sample_limit": 8,
    }


def record_failure(tracker, fail_type, identifier, snippet, pbar=None):
    """Record a failure event and emit inline diagnostics.

    Args:
        tracker: dict returned by :func:`create_failure_tracker`.
        fail_type: ``"api_error"`` or ``"failed_extraction"``.
        identifier: a persona/twin ID or other label.
        snippet: short text excerpt for diagnosis.
        pbar: optional tqdm bar for ``tqdm.write`` output.
    """
    tracker["stats"][fail_type] += 1
    sample_store = tracker["samples"][fail_type]
    if len(sample_store) < tracker["sample_limit"]:
        sample_store.append((identifier, snippet))

    n_this_type = tracker["stats"][fail_type]
    writer = tqdm.write if pbar is not None else print

    if n_this_type <= 3:
        writer(
            f"  [{fail_type.upper()} #{n_this_type}] "
            f"id={identifier} | {snippet}"
        )
    elif n_this_type % 50 == 0:
        total_fail = tracker["stats"]["api_error"] + tracker["stats"]["failed_extraction"]
        total_seen = tracker["stats"]["success"] + total_fail
        rate = 100.0 * total_fail / total_seen if total_seen else 0
        writer(
            f"  [FAILURE SPIKE] {total_fail} failures in "
            f"{total_seen} tasks ({rate:.1f}% failure rate). "
            f"Latest {fail_type}: {snippet[:120]}"
        )


def print_failure_summary(tracker, total_tasks):
    """Print the end-of-run failure summary with sample dump."""
    stats = tracker["stats"]
    fail_total = stats["failed_extraction"] + stats["api_error"]

    print(f"\nTotal tasks processed : {total_tasks}")
    print(f"Success               : {stats['success']}")
    print(f"Failed extraction     : {stats['failed_extraction']}")
    print(f"API errors            : {stats['api_error']}")
    if fail_total > 0 and total_tasks > 0:
        print(f"Overall failure rate  : {100 * fail_total / total_tasks:.1f}%")

    for ftype, samples in tracker["samples"].items():
        if not samples:
            continue
        print(f"\n--- {ftype.upper()} samples (first {len(samples)}) ---")
        for i, (pid, snippet) in enumerate(samples, 1):
            print(f"  [{i}] id={pid}")
            print(f"      {snippet}")


# ---------------------------------------------------------------------------
# Common CLI arguments
# ---------------------------------------------------------------------------


def add_common_args(parser):
    """Add the CLI flags shared by both experiment entry-points.

    Call this on an :class:`argparse.ArgumentParser` before adding any
    experiment-specific flags.
    """
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="Model name or friendly alias")
    parser.add_argument("--base-url", type=str, default=None, help="API base URL")
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY", ""), help="API key")
    parser.add_argument(
        "--provider", type=str, default="openai",
        choices=["openai", "bedrock", "openrouter", "vllm"],
        help="API provider",
    )
    parser.add_argument("--bedrock-region", type=str, default="us-east-1", help="AWS Bedrock region")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max completion tokens")
    parser.add_argument("--workers", "--max-workers", type=int, default=8, dest="max_workers", help="Max parallel workers")
    parser.add_argument("--db-path", type=str, default="results.db", help="Path to SQLite database")

    parser.add_argument("--thinking-model", action="store_true",
                        help="Enable thinking-model mode (extended tokens, judge extraction)")
    parser.add_argument("--thinking-max-tokens", type=int, default=8192,
                        help="Max tokens for thinking model generation")

    parser.add_argument("--judge-model", type=str, default=None, help="Model for LLM judge extraction")
    parser.add_argument("--judge-provider", type=str, default="openai",
                        choices=["openai", "bedrock", "openrouter", "vllm"],
                        help="Provider for the judge model")
    parser.add_argument("--judge-base-url", type=str, default=None, help="Base URL for judge model API")
    parser.add_argument("--judge-api-key", type=str, default=None, help="API key for judge model")
    parser.add_argument("--judge-bedrock-region", type=str, default="us-east-1",
                        help="AWS Bedrock region for judge model")
    parser.add_argument("--judge-only", action="store_true",
                        help="Skip data collection; only run judge extraction on stored raw responses")

    parser.add_argument("--flagged-personas-file", type=str, default=None,
                        help="Pickle file of integer persona IDs to skip")
    parser.add_argument("--personas-file", type=str, default=None,
                        help="Path to personas JSON file")
    parser.add_argument("--limit", type=int, default=None, help="Number of personas to process")


# ---------------------------------------------------------------------------
# Client setup from parsed args
# ---------------------------------------------------------------------------


def setup_clients(args):
    """Instantiate main and judge clients from parsed CLI arguments.

    Returns:
        ``(client, model, judge_client, judge_model)``
    """
    client, model = create_client(
        provider=args.provider,
        api_key=args.api_key,
        base_url=args.base_url,
        bedrock_region=args.bedrock_region,
        model=args.model,
    )

    judge_client = None
    judge_model = args.judge_model
    if args.judge_model or args.judge_only:
        judge_api_key = args.judge_api_key or args.api_key
        judge_client, judge_model = create_client(
            provider=args.judge_provider,
            api_key=judge_api_key,
            base_url=args.judge_base_url or args.base_url,
            bedrock_region=args.judge_bedrock_region,
            model=args.judge_model,
        )

    return client, model, judge_client, judge_model
