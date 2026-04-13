"""
check_persona_inconsistency.py

Scans persona JSON files for logically inconsistent attribute combinations
and saves a pickle of the flagged IDs.

Usage (from repository root):

    python -m analysis.check_persona_inconsistency
    python -m analysis.check_persona_inconsistency --dry-run
    python -m analysis.check_persona_inconsistency --output flagged.pkl

The script checks sampled_personas_2000.json for conflicts such as:
  - Child (0-17) who is Married/Divorced/Widowed
  - Child (0-17) who has children
  - Low education + high-skill occupation
  - Lower class + very high income (or vice-versa)
"""

import argparse
import json
import os
import pickle
import re

DEFAULT_PERSONAS = os.path.join("data", "sampled_personas_2000.json")
DEFAULT_OUTPUT = os.path.join("data", "flagged_personas.pkl")

RULES = [
    (
        "child_with_marital_status",
        "Child (0-17) + Married/Divorced/Widowed",
        {"Age_category", "Marital Status"},
        lambda a: (
            a.get("Age_category") == "Child (0-17)" and
            a.get("Marital Status") in ("Married", "Divorced", "Widowed")
        ),
    ),
    (
        "child_with_children",
        "Child (0-17) + Has Children = Yes",
        {"Age_category", "Has Children"},
        lambda a: (
            a.get("Age_category") == "Child (0-17)" and
            a.get("Has Children") == "Yes"
        ),
    ),
    (
        "low_education_high_skill_job",
        "Low Education + High-Skill Occupation",
        {"Education Level", "Occupation"},
        lambda a: (
            a.get("Education Level") in
                ("No Education", "Elementary School", "Middle School") and
            a.get("Occupation") in
                ("Teacher", "Doctor", "Engineer", "Lawyer", "Nurse")
        ),
    ),
    (
        "lower_class_high_income",
        "Lower Class + $150k+ Income",
        {"Social Class", "Annual Income"},
        lambda a: (
            a.get("Social Class") == "Lower class" and
            a.get("Annual Income") == "$150,000 and Above"
        ),
    ),
    (
        "upper_class_low_income",
        "Upper Class + <$9,999 Income",
        {"Social Class", "Annual Income"},
        lambda a: (
            a.get("Social Class") == "Upper class" and
            a.get("Annual Income") == "Less than $9,999"
        ),
    ),
    (
        "middle_class_extreme_income",
        "Middle Class + Extreme Income",
        {"Social Class", "Annual Income"},
        lambda a: (
            a.get("Social Class") == "Middle class" and
            a.get("Annual Income") in ("$150,000 and Above", "Less than $9,999")
        ),
    ),
]


def extract_attrs(profile):
    """Extract normalised attribute dict from a persona profile."""
    attrs = dict(profile.get("persona_attributes", {}))
    attrs["Age_category"] = attrs.get("Age", "")
    return attrs


def check_profile(attrs):
    """Return {rule_key: triggered} for applicable rules."""
    results = {}
    for key, _label, required, test_fn in RULES:
        if any(not attrs.get(attr) for attr in required):
            continue
        results[key] = bool(test_fn(attrs))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Check persona JSON files for logical inconsistencies."
    )
    parser.add_argument(
        "--personas-file", type=str, default=DEFAULT_PERSONAS,
        help=f"Path to personas JSON (default: {DEFAULT_PERSONAS})",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help=f"Output pickle file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print results but do not write pickle file",
    )
    args = parser.parse_args()

    print(f"Loading personas from {args.personas_file}...")
    with open(args.personas_file, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    print(f"Total profiles: {len(profiles)}")

    # Report rule applicability
    sample_attrs = extract_attrs(profiles[0]) if profiles else {}
    applicable_rules = []
    print("\nRule applicability (checked against first profile):")
    for key, label, required, _ in RULES:
        missing = [a for a in required if not sample_attrs.get(a)]
        if missing:
            print(f"  [SKIP] {label}  (missing: {', '.join(missing)})")
        else:
            print(f"  [OK]   {label}")
            applicable_rules.append(key)

    # Run checks
    flagged_ids = []
    condition_counts = {key: 0 for key, *_ in RULES}

    for profile in profiles:
        attrs = extract_attrs(profile)
        triggered = check_profile(attrs)
        if any(triggered.values()):
            flagged_ids.append(profile["id"])
            for key, fired in triggered.items():
                if fired:
                    condition_counts[key] += 1

    print(f"\nFlagged profiles: {len(flagged_ids)}")
    print("\nBreakdown by condition:")
    for key, label, required, _ in RULES:
        cnt = condition_counts[key]
        tag = "" if key in applicable_rules else "  [N/A]"
        print(f"  {label}: {cnt}{tag}")

    print(f"\nFlagged IDs: {flagged_ids}")

    if args.dry_run:
        print("\n[dry-run] No file written.")
    else:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "wb") as fh:
            pickle.dump(flagged_ids, fh)
        print(f"\nSaved -> {args.output}  ({len(flagged_ids)} IDs)")


if __name__ == "__main__":
    main()
