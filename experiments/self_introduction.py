#!/usr/bin/env python3
"""
Self-introduction collection.

Prompt every non-flagged persona to produce a detailed self-introduction.
Each persona yields N_SAMPLES independent responses.  Results are saved
incrementally to a JSONL file so runs can be resumed after interruption.

Supports two providers:
  - openrouter : calls the OpenRouter API (closed-source + open-source)
  - vllm       : calls a local vLLM OpenAI-compatible server

Usage (OpenRouter):
    python -m experiments.self_introduction \\
        --provider openrouter --api-key "$OPENROUTER_API_KEY" \\
        --models kimi-k2.5 claude-sonnet-4.6

Usage (vLLM):
    python -m experiments.self_introduction \\
        --provider vllm \\
        --base-url http://localhost:8123/v1 \\
        --models Qwen/Qwen3-4B-Instruct-2507

Usage (all OpenRouter models at once):
    python -m experiments.self_introduction --provider openrouter
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ── Configuration ────────────────────────────────────────────────────────────

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PERSONAS_FILE = "data/sampled_personas_2000.json"
FLAGGED_FILE = "data/flagged_personas.pkl"
OUTPUT_DIR = "self_introduction_results"

OPENROUTER_REGISTRY: dict[str, str] = {
    "claude-sonnet-4.6": "anthropic/claude-sonnet-4-6",
    "claude-haiku-4.5":  "anthropic/claude-haiku-4-5",
    "minimax-m2-her":    "minimax/minimax-m2-her",
    "minimax-m2.5":      "minimax/minimax-m2.5",
    "minimax-m2":        "minimax/minimax-m2",
    "kimi-k2.5":         "moonshotai/kimi-k2.5",
}

VLLM_MODELS: list[str] = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Neph0s/CoSER-Llama-3.1-8B",
]

SYSTEM_PROMPT = (
    "We will do a role-playing game. You will be given a persona description. "
    "Stay fully in character as that persona throughout your response."
)

USER_PROMPT_TEMPLATE = """{persona_description}
---
Please introduce yourself. Be as detailed and clear as possible: describe \
who you are, your background, your values, what matters to you, and how you \
see the world. Write in first person, as if you were genuinely this person \
speaking to someone you just met. Aim for a thorough, natural \
self-introduction (at least a few paragraphs)."""

N_SAMPLES_DEFAULT = 3
MAX_WORKERS_DEFAULT = 6
MAX_TOKENS = 1024
TEMPERATURE = 0.9


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_openai_client():
    """Lazily resolve the OpenAI client class (requires openai >= 1.0)."""
    try:
        from openai import OpenAI
        return OpenAI
    except (ImportError, AttributeError):
        raise ImportError(
            "openai >= 1.0.0 is required.  Run: pip install --upgrade openai"
        )


def load_personas(
    personas_file: str,
    flagged_file: str,
) -> list[dict]:
    """Load personas and exclude flagged IDs."""
    with open(personas_file, "r", encoding="utf-8") as f:
        all_personas = json.load(f)

    flagged_ids: set = set()
    if os.path.exists(flagged_file):
        with open(flagged_file, "rb") as f:
            flagged_ids = set(pickle.load(f))
        print(f"Loaded {len(flagged_ids)} flagged persona IDs to exclude")

    filtered = [p for p in all_personas if p.get("id") not in flagged_ids]
    print(
        f"Personas: {len(all_personas)} total, "
        f"{len(filtered)} after filtering"
    )
    return filtered


def load_completed(output_file: str) -> set[tuple[str, int, int]]:
    """
    Return set of ``(model, persona_id, sample_idx)`` already completed.

    Records whose response starts with ``[ERROR]`` are excluded so that
    failed tasks are retried on the next run.
    """
    done: set[tuple[str, int, int]] = set()
    n_errors_skipped = 0
    if not os.path.exists(output_file):
        return done

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("response", "").startswith("[ERROR]"):
                    n_errors_skipped += 1
                    continue
                done.add((
                    rec["model"], rec["persona_id"], rec["sample_idx"],
                ))
            except (json.JSONDecodeError, KeyError):
                continue

    if n_errors_skipped:
        print(
            f"  Skipped {n_errors_skipped} errored record(s) — "
            f"they will be retried."
        )
    return done


_write_lock = threading.Lock()


def append_result(output_file: str, record: dict):
    """Thread-safe append of one JSON record to a JSONL file."""
    with _write_lock:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def query_model(
    client,
    model_id: str,
    persona_description: str,
    max_retries: int = 5,
) -> tuple[str, dict]:
    """
    Call the chat-completions endpoint with exponential backoff.

    Returns ``(text, usage_dict)``.
    """
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": USER_PROMPT_TEMPLATE.format(
                            persona_description=persona_description,
                        ),
                    },
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            text = resp.choices[0].message.content or ""
            usage = {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(
                    resp.usage, "completion_tokens", 0,
                ),
            }
            return text, usage
        except Exception as e:
            err = str(e)
            if any(
                k in err.lower()
                for k in ("rate", "throttl", "429", "too many")
            ):
                wait = min(2 ** attempt * 2, 120)
                time.sleep(wait)
            elif attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return (
                    f"[ERROR] {err}",
                    {"prompt_tokens": 0, "completion_tokens": 0},
                )

    return (
        "[ERROR] max retries",
        {"prompt_tokens": 0, "completion_tokens": 0},
    )


def friendly_name(model_id: str) -> str:
    """Derive a short filename-safe label from a HuggingFace model ID."""
    return model_id.rsplit("/", 1)[-1]


# ── Progress tracker ─────────────────────────────────────────────────────────

class ProgressTracker:
    """Thread-safe progress tracker with periodic ETA reports."""

    def __init__(self, total: int):
        self.total = total
        self.done = 0
        self.errors = 0
        self.lock = threading.Lock()
        self.start = time.time()

    def update(self, is_error: bool = False):
        with self.lock:
            self.done += 1
            if is_error:
                self.errors += 1
            if self.done % 50 == 0 or self.done == self.total:
                elapsed = time.time() - self.start
                rate = self.done / elapsed if elapsed > 0 else 0
                eta = (self.total - self.done) / rate if rate > 0 else 0
                print(
                    f"  [{self.done:>6}/{self.total}]  "
                    f"{rate:.1f} req/s  ETA {eta / 60:.0f}m  "
                    f"errors={self.errors}"
                )


# ── Core loop (one model at a time) ─────────────────────────────────────────

def run_model(
    client,
    model_name: str,
    model_id: str,
    personas: list[dict],
    n_samples: int,
    output_dir: str,
    max_workers: int,
):
    """Collect self-introduction responses for a single model."""
    output_file = os.path.join(
        output_dir, f"introductions_{model_name}.jsonl",
    )
    completed = load_completed(output_file)

    print(f"\n{'=' * 70}")
    print(f"Model: {model_name}  ({model_id})")
    print(f"  Output: {output_file}")
    print(f"  Samples per persona : {n_samples}")

    expected_total = len(personas) * n_samples
    print(
        f"  Expected total tasks: {expected_total}  "
        f"({len(personas)} personas x {n_samples} samples)"
    )
    print(f"  Already completed   : {len(completed)}")

    tasks: list[tuple] = []
    for persona in personas:
        pid = persona["id"]
        desc = persona["description"]
        for s in range(n_samples):
            if (model_name, pid, s) in completed:
                continue
            tasks.append((pid, persona.get("persona", ""), desc, s))

    if not tasks:
        print("  All done — skipping.")
        return

    print(f"  Remaining: {len(tasks)}")
    tracker = ProgressTracker(len(tasks))

    def _worker(pid, plabel, desc, s_idx):
        text, usage = query_model(client, model_id, desc)
        is_err = text.startswith("[ERROR]")
        if is_err:
            tracker.update(is_error=True)
            return
        record = {
            "model": model_name,
            "model_id": model_id,
            "persona_id": pid,
            "persona_label": plabel,
            "sample_idx": s_idx,
            "response": text,
            "usage": usage,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        append_result(output_file, record)
        tracker.update(is_error=False)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_worker, *t) for t in tasks]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:
                print(f"  [FATAL] {exc}")

    print(f"  Finished model {model_name}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Collect self-introduction responses from LLMs",
    )
    parser.add_argument(
        "--provider",
        default="openrouter",
        choices=["openrouter", "vllm"],
        help="API provider",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENROUTER_API_KEY", ""),
        help="API key (OpenRouter) or dummy for vLLM",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help=(
            "Base URL for the API (auto-set for openrouter; "
            "required for vllm, e.g. http://localhost:8123/v1)"
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Model names.  For openrouter: friendly names from "
            "OPENROUTER_REGISTRY.  For vllm: HuggingFace model IDs."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=N_SAMPLES_DEFAULT,
        help="Number of responses per persona per model",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=MAX_WORKERS_DEFAULT,
        help="Concurrent API threads",
    )
    parser.add_argument("--personas-file", default=PERSONAS_FILE)
    parser.add_argument("--flagged-file", default=FLAGGED_FILE)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap number of personas (for testing)",
    )
    args = parser.parse_args()

    # ── Resolve provider settings ────────────────────────────────────────
    if args.provider == "openrouter":
        if not args.api_key:
            raise SystemExit(
                "Provide --api-key or set OPENROUTER_API_KEY "
                "for openrouter provider."
            )
        base_url = args.base_url or OPENROUTER_BASE_URL
        model_names = args.models or list(OPENROUTER_REGISTRY.keys())
        models = []
        for name in model_names:
            mid = OPENROUTER_REGISTRY.get(name, name)
            models.append((name, mid))
    else:  # vllm
        if not args.base_url:
            raise SystemExit(
                "--base-url is required for vllm provider "
                "(e.g. http://localhost:8123/v1)"
            )
        base_url = args.base_url
        args.api_key = args.api_key or "dummy-key"
        model_names = args.models or VLLM_MODELS
        models = [(friendly_name(m), m) for m in model_names]

    os.makedirs(args.output_dir, exist_ok=True)
    OpenAI = _get_openai_client()
    client = OpenAI(api_key=args.api_key, base_url=base_url)

    personas = load_personas(args.personas_file, args.flagged_file)
    if args.limit:
        personas = personas[: args.limit]

    print(f"\nProvider: {args.provider}  Base URL: {base_url}")
    print(f"Models ({len(models)}):")
    for name, mid in models:
        print(f"  {name:40s} -> {mid}")

    for model_name, model_id in models:
        run_model(
            client, model_name, model_id, personas,
            args.samples, args.output_dir, args.max_workers,
        )

    print(f"\nAll done.  Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
