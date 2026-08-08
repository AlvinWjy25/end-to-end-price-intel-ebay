"""
ingest_v2_sample_probe.py
Diagnostic / sampling script -- NOT part of the production pipeline.
Does NOT insert to Postgres.

Purpose:
Pull a small sample of items per keyword (reusing the same search logic as
ingest.py) and inspect how consistently eBay's getItem response exposes
localizedAspects (item specifics) -- especially fields like "Features",
"Issue Number", and categoryPath -- so we can decide whether these fields
are reliable enough to replace/augment title-regex volume parsing.

Usage:
    python ingest_v2_sample_probe.py

Output:
    ./debug_output/raw/<item_id_sanitized>.json   -- full raw response per item
    ./debug_output/aspect_summary.json            -- aggregated coverage summary
    Also prints a human-readable summary to stdout.
"""

import os
import json
import time
import requests
from dotenv import load_dotenv
from collections import Counter

load_dotenv()

EBAY_CLIENT_ID = os.getenv("EBAY_CLIENT_ID")
EBAY_CLIENT_SECRET = os.getenv("EBAY_CLIENT_SECRET")

EBAY_OAUTH_URL = "https://api.ebay.com/identity/v1/oauth2/token"
EBAY_SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"
EBAY_ITEM_URL = "https://api.ebay.com/buy/browse/v1/item"

OUTPUT_DIR = "debug_output"
RAW_DIR = os.path.join(OUTPUT_DIR, "raw")

SAMPLE_PER_KEYWORD = 8 # small on purpose -- this is a probe, not a full ingest
REQUEST_DELAY = 0.15


def get_access_token() -> str:
    if not EBAY_CLIENT_ID or not EBAY_CLIENT_SECRET:
        raise ValueError("EBAY_CLIENT_ID / EBAY_CLIENT_SECRET not found in .env")

    response = requests.post(
        EBAY_OAUTH_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"},
        auth=(EBAY_CLIENT_ID, EBAY_CLIENT_SECRET),
        timeout=10,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def search_item_ids(access_token: str, keyword: str, limit: int) -> list[str]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
    }
    params = {"q": keyword, "limit": limit}

    response = requests.get(EBAY_SEARCH_URL, headers=headers, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    items = data.get("itemSummaries", [])
    return [item["itemId"] for item in items if "itemId" in item]


def get_item_raw(access_token: str, item_id: str) -> dict | None:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
    }
    url = f"{EBAY_ITEM_URL}/{item_id}"
    response = requests.get(url, headers=headers, timeout=10)

    if response.status_code != 200:
        return None

    return response.json()


def aspects_to_dict(localized_aspects: list | None) -> dict:
    """Convert eBay's [{name, value, type}, ...] into a flat {name: value} dict."""
    if not localized_aspects:
        return {}
    return {a.get("name"): a.get("value") for a in localized_aspects if a.get("name")}


def main():
    keywords = [
        "re:zero light novel",
        "the angel next door light novel",
        "makine too many losing heroines light novel",
        "86 eighty six light novel",
        "gimai seikatsu light novel",
        "love unseen beneath the clear night sky light novel",
        "love unseen beneath the radiant night sky:volume",
        "even a replica can fall in love light novel",
        "classroom of the elite light novel",
        "mushoku tensei light novel",
        "overlord light novel",
        "konosuba light novel",
        "sword art online light novel",
        "spy classroom light novel",
        "rascal does not dream of bunny girl senpai light novel",
        "the eminence in shadow light novel",
        "bofuri light novel",
        "ascendance of a bookworm light novel",
    ]

    os.makedirs(RAW_DIR, exist_ok=True)

    token = get_access_token()
    print("Access token obtained.\n")

    # Stage 1: gather a small sample of item_ids per keyword
    sample_item_ids = []
    for kw in keywords:
        ids = search_item_ids(token, kw, limit=SAMPLE_PER_KEYWORD)
        sample_item_ids.extend(ids)
        print(f"  - '{kw}': {len(ids)} sampled")

    sample_item_ids = list(dict.fromkeys(sample_item_ids))  # dedupe, preserve order
    print(f"\nTotal unique sampled item_id: {len(sample_item_ids)}\n")

    # Stage 2: pull getItem detail for each sampled item, record aspect coverage
    total = 0
    has_localized_aspects = 0
    has_features = 0
    has_issue_number = 0
    has_series_title = 0
    aspect_name_counter = Counter()
    category_path_counter = Counter()
    examples_with_features = []
    examples_without_aspects = []

    for i, item_id in enumerate(sample_item_ids, start=1):
        data = get_item_raw(token, item_id)
        time.sleep(REQUEST_DELAY)

        if data is None:
            continue

        total += 1

        # Save full raw response for later manual inspection if needed
        safe_name = item_id.replace("|", "_").replace("/", "_")
        with open(os.path.join(RAW_DIR, f"{safe_name}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        aspects = aspects_to_dict(data.get("localizedAspects"))
        category_path = data.get("categoryPath", "")

        if aspects:
            has_localized_aspects += 1
            aspect_name_counter.update(aspects.keys())
        else:
            examples_without_aspects.append({"item_id": item_id, "title": data.get("title")})

        if "Features" in aspects:
            has_features += 1
            examples_with_features.append({
                "item_id": item_id,
                "title": data.get("title"),
                "Features": aspects["Features"],
            })

        if "Issue Number" in aspects:
            has_issue_number += 1

        if "Series Title" in aspects:
            has_series_title += 1

        if category_path:
            category_path_counter[category_path] += 1

        if i % 10 == 0:
            print(f"  ...probed {i}/{len(sample_item_ids)}")

    # Stage 3: print + save summary
    summary = {
        "total_items_probed": total,
        "pct_with_localized_aspects": round(100 * has_localized_aspects / total, 1) if total else 0,
        "pct_with_features_field": round(100 * has_features / total, 1) if total else 0,
        "pct_with_issue_number_field": round(100 * has_issue_number / total, 1) if total else 0,
        "pct_with_series_title_field": round(100 * has_series_title / total, 1) if total else 0,
        "most_common_aspect_names": aspect_name_counter.most_common(20),
        "category_paths_seen": category_path_counter.most_common(10),
        "sample_examples_with_features": examples_with_features[:10],
        "sample_examples_without_aspects": examples_without_aspects[:10],
    }

    with open(os.path.join(OUTPUT_DIR, "aspect_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total items probed:            {summary['total_items_probed']}")
    print(f"Has localizedAspects at all:   {summary['pct_with_localized_aspects']}%")
    print(f"Has 'Features' field:          {summary['pct_with_features_field']}%")
    print(f"Has 'Issue Number' field:      {summary['pct_with_issue_number_field']}%")
    print(f"Has 'Series Title' field:      {summary['pct_with_series_title_field']}%")
    print(f"\nMost common aspect names across sample:")
    for name, count in summary["most_common_aspect_names"]:
        print(f"  {name}: {count}")
    print(f"\nCategory paths seen:")
    for path, count in summary["category_paths_seen"]:
        print(f"  ({count}x) {path}")
    print(f"\nFull summary saved -> {os.path.join(OUTPUT_DIR, 'aspect_summary.json')}")
    print(f"Full raw responses saved -> {RAW_DIR}/")


if __name__ == "__main__":
    main()