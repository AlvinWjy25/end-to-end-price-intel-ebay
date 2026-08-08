"""
inspect_unit_style_fields.py
Diagnostic script -- reads the raw JSON files already saved by
ingest_v2_sample_probe.py (in debug_output/raw/) and reports the
distinct values seen for specific aspect fields, so we can judge
whether they're useful signals for volume-count / single-vs-set logic.

Usage:
    python inspect_unit_style_fields.py
"""

import os
import json
from collections import Counter, defaultdict

RAW_DIR = os.path.join("debug_output", "raw")
FIELDS_TO_CHECK = ["Unit of Sale", "Style", "Format", "Genre", "Intended Audience"]


def aspects_to_dict(localized_aspects):
    if not localized_aspects:
        return {}
    return {a.get("name"): a.get("value") for a in localized_aspects if a.get("name")}


def main():
    if not os.path.isdir(RAW_DIR):
        print(f"'{RAW_DIR}' not found. Run ingest_v2_sample_probe.py first.")
        return

    value_counters = {field: Counter() for field in FIELDS_TO_CHECK}
    # Also keep a few example titles per value, so we can sanity-check meaning
    examples = {field: defaultdict(list) for field in FIELDS_TO_CHECK}

    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".json")]
    print(f"Inspecting {len(files)} saved raw responses...\n")

    for fname in files:
        with open(os.path.join(RAW_DIR, fname), "r", encoding="utf-8") as f:
            data = json.load(f)

        aspects = aspects_to_dict(data.get("localizedAspects"))
        title = data.get("title", "")

        for field in FIELDS_TO_CHECK:
            value = aspects.get(field)
            if value:
                value_counters[field][value] += 1
                if len(examples[field][value]) < 3:
                    examples[field][value].append(title)

    for field in FIELDS_TO_CHECK:
        counter = value_counters[field]
        print(f"{'=' * 70}")
        print(f"Field: {field}   (present in {sum(counter.values())} of {len(files)} items)")
        print(f"{'=' * 70}")
        if not counter:
            print("  (never present in this sample)\n")
            continue

        for value, count in counter.most_common(15):
            print(f"  [{count:>3}x] {value}")
            for ex_title in examples[field][value]:
                print(f"           e.g. \"{ex_title}\"")
        print()


if __name__ == "__main__":
    main()