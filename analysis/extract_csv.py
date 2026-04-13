"""
Extract response matrices from experiment databases into organized CSV files.

Supports both DB schemas:
  - Digital twin (BFI):    table 'bfi_responses'  with (twin_id, question_id, score, profile_json)
  - Moral reasoning:       table 'results'        with (profile_json, scenario, score, prompt)

Each DB produces one CSV file where:
  - Rows    = personas (one per unique persona)
  - Columns = persona attributes  +  response scores (one column per question/scenario)

Output structure:
  <output_root>/
    <dataset>_<model_name>/
      response_matrix.csv

Usage:
  python extract_csv.py --db-pattern "results_moral_*.db" --dataset moral_reasoning
  python extract_csv.py --db-pattern "results_replicate_humans_*_bfi.db" --dataset digital_twin
  python extract_csv.py --db-pattern "results_moral_*.db"   # auto-detects dataset

Flagged-persona filtering
-------------------------
Two complementary mechanisms keep flagged personas out of the output CSVs:

  1. DUMMY_SENTINEL filter  (new DBs, >= the dummy-injection era)
     Rows where raw_response = '__DUMMY_FLAGGED_PERSONA__' were written by
     inject_dummy_moral_responses() / inject_dummy_bfi_responses() as cheap
     placeholders so the API was never called for flagged personas.
     These are filtered at the SQL level.

  2. ID-based filter  (old DBs, pre-dummy-injection, or missing raw_response column)
     For databases built before the dummy-injection trick existed the flagged
     personas may have real responses or simply no raw_response column at all
     (causing the SQL WHERE clause to fail).  In that case we fall back to
     loading a flagged_personas.pkl file and dropping any persona whose integer
     id appears in those sets.
"""

import sqlite3
import json
import os
import glob
import pickle
import argparse
import pandas as pd
from collections import defaultdict

DUMMY_SENTINEL = "__DUMMY_FLAGGED_PERSONA__"

DEFAULT_FLAGGED_PKLS = [
    "data/flagged_personas.pkl",
]


def load_flagged_ids(*pkl_paths: str) -> set:
    """
    Load integer persona IDs to exclude from one or more pickle files.
    Missing files are silently skipped.  Returns a (possibly empty) set of ints.
    """
    ids: set = set()
    for path in pkl_paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, "rb") as fh:
            batch = pickle.load(fh)
        ids.update(int(x) for x in batch)
        print(f"  Loaded {len(batch):,} flagged IDs from {path}")
    return ids


def _has_column(cursor, table: str, column: str) -> bool:
    """Return True if *column* exists in *table* of the currently open DB."""
    rows = cursor.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == column for r in rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_model_name(db_path):
    base = os.path.basename(db_path).replace(".db", "")
    for prefix in ("results_moral_", "results_replicate_humans_", "results_"):
        if base.startswith(prefix):
            base = base[len(prefix):]
    for suffix in ("_bfi",):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base


def detect_db_type(db_path):
    """Return 'digital_twin' or 'moral_reasoning' based on tables present."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    tables = [r[0] for r in cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    conn.close()

    if "bfi_responses" in tables:
        return "digital_twin"
    if "results" in tables:
        return "moral_reasoning"
    raise ValueError(f"Unknown DB schema in {db_path}. Tables: {tables}")


def discover_db_files(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files matched pattern: {pattern}")
    return files


def _flatten_profile(profile):
    """Flatten a persona profile dict into a flat key-value mapping."""
    flat = {}
    if not isinstance(profile, dict):
        return flat
    for k, v in profile.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                flat[k2] = v2
        else:
            flat[k] = v
    return flat


# ---------------------------------------------------------------------------
# Digital Twin (BFI) Extraction
# ---------------------------------------------------------------------------

def extract_digital_twin(db_path, flagged_ids: set = None):
    """
    Returns a DataFrame: rows = personas, columns = profile attrs + BFI questions.

    Filtering strategy (applied in order):
      1. DUMMY_SENTINEL SQL filter -- fast, works for DBs built with dummy injection.
      2. ID-based filter via flagged_ids -- fallback for old DBs without raw_response
         column or written before the dummy-injection era.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM bfi_responses")
    total_rows = cursor.fetchone()[0]

    if _has_column(cursor, "bfi_responses", "raw_response"):
        cursor.execute(
            "SELECT twin_id, question_id, score, profile_json FROM bfi_responses "
            "WHERE raw_response IS NULL OR raw_response != ?",
            (DUMMY_SENTINEL,),
        )
        rows = cursor.fetchall()
        n_sentinel = total_rows - len(rows)
        if n_sentinel:
            print(f"  Filtered {n_sentinel:,} dummy-sentinel rows from {total_rows:,} total")
    else:
        print(f"  Note: 'raw_response' column not found -- will filter by flagged IDs instead.")
        cursor.execute(
            "SELECT twin_id, question_id, score, profile_json FROM bfi_responses"
        )
        rows = cursor.fetchall()

    conn.close()

    personas = defaultdict(lambda: {"scores": {}, "profile": {}})
    for twin_id, question_id, score, profile_json in rows:
        personas[twin_id]["scores"][str(question_id)] = score
        if not personas[twin_id]["profile"]:
            try:
                personas[twin_id]["profile"] = json.loads(profile_json)
            except Exception:
                pass

    if flagged_ids:
        before = len(personas)
        personas = {
            tid: data for tid, data in personas.items()
            if not (str(tid).lstrip("-").isdigit() and int(tid) in flagged_ids)
        }
        n_id_dropped = before - len(personas)
        if n_id_dropped:
            print(f"  Filtered {n_id_dropped:,} flagged personas by ID "
                  f"(from pkl) out of {before:,} personas")

    records = []
    all_qids = set()
    for tid, data in personas.items():
        all_qids.update(data["scores"].keys())

    qid_list = sorted(all_qids, key=lambda x: int(x) if x.isdigit() else x)

    for tid, data in personas.items():
        flat_profile = _flatten_profile(data["profile"])
        row = {"persona_id": tid}
        row.update(flat_profile)
        for qid in qid_list:
            row[f"BFI_Q{qid}"] = data["scores"].get(qid, None)
        records.append(row)

    df = pd.DataFrame(records)
    n_personas = len(df)
    n_questions = len(qid_list)
    print(f"  Extracted {n_personas} personas x {n_questions} BFI questions")
    return df


# ---------------------------------------------------------------------------
# Moral Reasoning Extraction
# ---------------------------------------------------------------------------

def extract_moral_reasoning(db_path, flagged_ids: set = None):
    """
    Returns a DataFrame: rows = personas, columns = profile attrs + scenarios.

    Filtering strategy (applied in order):
      1. DUMMY_SENTINEL SQL filter -- fast, works for DBs built with dummy injection.
      2. ID-based filter via flagged_ids -- fallback for old DBs without raw_response
         column or written before the dummy-injection era.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM results")
    total_rows = cursor.fetchone()[0]

    if _has_column(cursor, "results", "raw_response"):
        cursor.execute(
            "SELECT profile_json, scenario, score FROM results "
            "WHERE raw_response IS NULL OR raw_response != ?",
            (DUMMY_SENTINEL,),
        )
        rows = cursor.fetchall()
        n_sentinel = total_rows - len(rows)
        if n_sentinel:
            print(f"  Filtered {n_sentinel:,} dummy-sentinel rows from {total_rows:,} total")
    else:
        print(f"  Note: 'raw_response' column not found -- will filter by flagged IDs instead.")
        cursor.execute("SELECT profile_json, scenario, score FROM results")
        rows = cursor.fetchall()

    conn.close()

    persona_map = {}
    scenario_set = set()

    for profile_json, scenario, score in rows:
        try:
            profile = json.loads(profile_json)
        except Exception:
            profile = {}

        flat = _flatten_profile(profile)
        p_key = json.dumps(flat, sort_keys=True)

        if p_key not in persona_map:
            persona_map[p_key] = {"profile": flat, "scores": {}, "_id": profile.get("id")}
        persona_map[p_key]["scores"][scenario] = score
        scenario_set.add(scenario)

    if flagged_ids:
        before = len(persona_map)
        persona_map = {
            k: v for k, v in persona_map.items()
            if v.get("_id") is None or int(v["_id"]) not in flagged_ids
        }
        n_id_dropped = before - len(persona_map)
        if n_id_dropped:
            print(f"  Filtered {n_id_dropped:,} flagged personas by ID "
                  f"(from pkl) out of {before:,} personas")

    scenario_list = sorted(scenario_set)
    short_names = {s: f"scenario_{i + 1}" for i, s in enumerate(scenario_list)}

    records = []
    for p_key, data in persona_map.items():
        row = {"persona_id": hash(p_key) % (10 ** 10)}
        row.update(data["profile"])
        for scenario in scenario_list:
            row[short_names[scenario]] = data["scores"].get(scenario, None)
        records.append(row)

    df = pd.DataFrame(records)
    n_personas = len(df)
    n_scenarios = len(scenario_list)
    print(f"  Extracted {n_personas} personas x {n_scenarios} scenarios")

    scenario_df = pd.DataFrame([
        {"column_name": short_names[s], "scenario_text": s}
        for s in scenario_list
    ])

    return df, scenario_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract persona x question/scenario CSV files from experiment DBs")
    parser.add_argument("--db-pattern", type=str, default=None,
                        help="Glob pattern for DB files")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Explicit list of DB paths")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["digital_twin", "moral_reasoning"],
                        help="Dataset type (auto-detected if omitted)")
    parser.add_argument("--output", type=str, default="csv_exports",
                        help="Root output directory")
    parser.add_argument(
        "--flagged-files", nargs="*",
        default=DEFAULT_FLAGGED_PKLS,
        help=(
            "Pickle files containing integer persona IDs to exclude. "
            "Used as a fallback for old DBs that predate the dummy-injection "
            "trick (no raw_response column or no DUMMY_SENTINEL rows). "
            "Defaults: %(default)s"
        ),
    )
    args = parser.parse_args()

    model_dbs = list(args.models) if args.models else []
    if args.db_pattern:
        model_dbs.extend(discover_db_files(args.db_pattern))

    seen = set()
    unique_dbs = []
    for p in model_dbs:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            unique_dbs.append(p)
    model_dbs = unique_dbs

    if not model_dbs:
        print("No DB files specified. Use --db-pattern or --models.")
        return

    flagged_ids = load_flagged_ids(*(args.flagged_files or []))
    if flagged_ids:
        print(f"Total flagged persona IDs loaded: {len(flagged_ids):,}")
    else:
        print("No flagged persona ID files found -- ID-based filter disabled.")

    for db_path in model_dbs:
        model_name = extract_model_name(db_path)

        if args.dataset:
            db_type = args.dataset
        else:
            db_type = detect_db_type(db_path)

        dir_name = f"{db_type}_{model_name}"
        out_dir = os.path.join(args.output, dir_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\nProcessing: {db_path}")
        print(f"  Type: {db_type}  |  Model: {model_name}")

        if db_type == "digital_twin":
            df = extract_digital_twin(db_path, flagged_ids=flagged_ids)
            csv_path = os.path.join(out_dir, f"{dir_name}_response_matrix.csv")
            df.to_csv(csv_path, index=False)
            print(f"  Saved: {csv_path}")

        elif db_type == "moral_reasoning":
            df, scenario_lookup = extract_moral_reasoning(db_path, flagged_ids=flagged_ids)
            csv_path = os.path.join(out_dir, f"{dir_name}_response_matrix.csv")
            df.to_csv(csv_path, index=False)
            print(f"  Saved: {csv_path}")

            lookup_path = os.path.join(out_dir, f"{dir_name}_scenario_lookup.csv")
            scenario_lookup.to_csv(lookup_path, index=False)
            print(f"  Saved: {lookup_path}  (maps column names -> full scenario text)")

        else:
            print(f"  ERROR: Unknown db type '{db_type}' for {db_path}")

    print(f"\nAll CSVs saved under {args.output}/")
    print("Done!")


if __name__ == "__main__":
    main()
