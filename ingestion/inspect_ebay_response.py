"""
inspect_ebay_response.py
Diagnostic script -- NOT part of the production pipeline.

Purpose:
Dump raw getItem API response to JSON files so we can inspect what fields
eBay actually exposes (e.g. localizedAspects / item specifics), without
touching the existing ingest.py or raw.ebay_listings table.

Usage:
    python inspect_ebay_response.py

Output:
    ./debug_output/<item_id_sanitized>.json  (one file per sampled item)
    Also prints top-level keys and localizedAspects to stdout for quick scan.
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

EBAY_CLIENT_ID = os.getenv("EBAY_CLIENT_ID")
EBAY_CLIENT_SECRET = os.getenv("EBAY_CLIENT_SECRET")

EBAY_OAUTH_URL = "https://api.ebay.com/identity/v1/oauth2/token"
EBAY_ITEM_URL = "https://api.ebay.com/buy/browse/v1/item"

OUTPUT_DIR = "debug_output"


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


def get_item_raw(access_token: str, item_id: str) -> dict | None:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
    }
    url = f"{EBAY_ITEM_URL}/{item_id}"
    response = requests.get(url, headers=headers, timeout=10)

    if response.status_code != 200:
        print(f"  [SKIP] {item_id} -> status {response.status_code}")
        return None

    return response.json()


def inspect(item_id: str, data: dict) -> None:
    print(f"\n{'=' * 70}")
    print(f"item_id: {item_id}")
    print(f"title: {data.get('title')}")
    print(f"{'=' * 70}")

    print(f"\nAll top-level keys in response:")
    print(sorted(data.keys()))

    aspects = data.get("localizedAspects")
    print(f"\nlocalizedAspects ({'FOUND' if aspects else 'NOT FOUND / EMPTY'}):")
    print(json.dumps(aspects, indent=2, ensure_ascii=False))

    # A few other fields that might help distinguish single vs set/lot listings
    for key in ["itemGroupType", "itemGroupHref", "quantityLimitPerBuyer",
                "estimatedAvailabilities", "unitPrice", "unitPricingMeasure"]:
        if key in data:
            print(f"\n{key}: {json.dumps(data[key], indent=2, ensure_ascii=False)}")


def main():
    # Fill in the item_ids you want to inspect.
    # Suggestion: include the problematic ones from the DBeaver screenshots
    # (multi-dash volume range, and the "loose set single" ambiguous case),
    # plus 1-2 "normal" single-volume items as a baseline comparison.
    sample_item_ids = [
        "v1|307070835585|0",   # "Classroom Of The Elite Complete Volume 1-11.5 Light Novel Set"
        # TODO: replace/add with the actual item_id for
        # "Classroom Of The Elite Light Novel Volumes 1-2-3-4"
        # "Too Many Losing Heroines! English Light Novel Volume 1-2 Loose Set Single"
        # and one plain single-volume listing, e.g. "Overlord, Vol. 6 (light Novel)"
    ]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    token = get_access_token()
    print("Access token obtained.\n")

    for item_id in sample_item_ids:
        data = get_item_raw(token, item_id)
        if data is None:
            continue

        inspect(item_id, data)

        # Save full raw response for later reference
        safe_name = item_id.replace("|", "_").replace("/", "_")
        out_path = os.path.join(OUTPUT_DIR, f"{safe_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\nFull response saved -> {out_path}")


if __name__ == "__main__":
    main()