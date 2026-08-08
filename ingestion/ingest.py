"""
ingest.py
Fetch listing data from eBay Browse API, then insert to Postgres (schema: raw).

Process:
1. Authenticate to eBay using OAuth 2.0 Client Credentials flow -> get access_token
2. Call item_summary/search endpoint with keyword -> get list of item_id
3. For each item_id, call getItem endpoint -> get full description (not just shortDescription)
   This is important because disclaimers such as "NOT Original / reprint edition" are usually found
   in the free description written by the seller, not in the auto-generated shortDescription.
4. Parse & strip HTML from description -> get relevant field
5. Insert to tabel raw.ebay_listings
"""

import os
import time
import requests
import psycopg2
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from datetime import datetime

# ---------------------------------------------------------
# 1. Load credentials from .env
# ---------------------------------------------------------
load_dotenv()

EBAY_CLIENT_ID = os.getenv("EBAY_CLIENT_ID")
EBAY_CLIENT_SECRET = os.getenv("EBAY_CLIENT_SECRET")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "price_intelligence")
DB_USER = os.getenv("DB_USER", "alvin")
DB_PASSWORD = os.getenv("DB_PASSWORD")

EBAY_OAUTH_URL = "https://api.ebay.com/identity/v1/oauth2/token"
EBAY_SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"
EBAY_ITEM_URL = "https://api.ebay.com/buy/browse/v1/item"


# ---------------------------------------------------------
# 2. Get OAuth access token
# ---------------------------------------------------------
def get_access_token() -> str:
    """
    eBay Browse API uses OAuth 2.0 Client Credentials flow.
    This token is temporary (usually ~2 hours), so it is generated
    anew each time the script is run -- no need to store permanently.
    """
    if not EBAY_CLIENT_ID or not EBAY_CLIENT_SECRET:
        raise ValueError(
            "EBAY_CLIENT_ID / EBAY_CLIENT_SECRET not filled in .env. "
            "Fill in first after eBay Developer Account is approved."
        )

    response = requests.post(
        EBAY_OAUTH_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"},
        auth=(EBAY_CLIENT_ID, EBAY_CLIENT_SECRET),
        timeout=10,
    )
    response.raise_for_status()
    return response.json()["access_token"]


# ---------------------------------------------------------
# 3. Search listing by keyword -> return list item_id
# ---------------------------------------------------------
def search_item_ids(access_token: str, keyword: str, limit: int = 30) -> list[str]:
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


# ---------------------------------------------------------
# 4. Get full detail of 1 item (including full description)
# ---------------------------------------------------------
def get_item_detail(access_token: str, item_id: str) -> dict | None:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
    }
    url = f"{EBAY_ITEM_URL}/{item_id}"

    response = requests.get(url, headers=headers, timeout=10)

    # Some items may have been delisted/no longer available
    # between the time search was called and getItem was called -- just skip, don't crash.
    if response.status_code != 200:
        return None

    return response.json()


def strip_html(raw_html: str | None) -> str | None:
    """eBay description is raw HTML (contains <b>, <br>, etc.).
    We strip it to get clean plain text for storage and parsing."""
    if not raw_html:
        return None
    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return text


# ---------------------------------------------------------
# 5. Parse response getItem to tuple
# ---------------------------------------------------------
def parse_item(item: dict) -> tuple:
    item_id = item.get("itemId")
    title = item.get("title")

    price_info = item.get("price", {})
    price = price_info.get("value")
    currency = price_info.get("currency")

    condition = item.get("condition")

    location_info = item.get("itemLocation", {})
    seller_location = location_info.get("country")

    # eBay description is raw HTML -> must be stripped first.
    description_raw = item.get("description")
    description_clean = strip_html(description_raw)

    return (item_id, title, price, currency, condition, seller_location, description_clean)


# ---------------------------------------------------------
# 5. Insert batch to Postgres
# ---------------------------------------------------------
def insert_listings(rows: list[tuple]) -> int:
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
    )
    cur = conn.cursor()

    insert_query = """
        INSERT INTO raw.ebay_listings (item_id, title, price, currency, condition, seller_location, description)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    cur.executemany(insert_query, rows)
    conn.commit()

    inserted_count = cur.rowcount
    cur.close()
    conn.close()

    return inserted_count


# ---------------------------------------------------------
# 6. Main
# ---------------------------------------------------------
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
        "ascendance of a bookworm light novel"
    ]

    SEARCH_LIMIT = 150  # Item/keyword search limitation
    REQUEST_DELAY = 0.15  # preventing throttle and spam detection

    print(f"[{datetime.now()}] Starting ingestion process...")

    token = get_access_token()
    print("Access token successfully obtained.")

    # Stage 1: Gather all item_id from each keyword
    all_item_ids = []
    for kw in keywords:
        ids = search_item_ids(token, kw, limit=SEARCH_LIMIT)
        all_item_ids.extend(ids)
        print(f"  - '{kw}': {len(ids)} item_id found")

    # Deduplicate
    all_item_ids = list(dict.fromkeys(all_item_ids))
    print(f"Total unique item_id: {len(all_item_ids)}")

    # Stage 2: Get full detail of each item
    all_rows = []
    skipped = 0
    for i, item_id in enumerate(all_item_ids, start=1):
        detail = get_item_detail(token, item_id)
        if detail is None:
            skipped += 1
            continue

        all_rows.append(parse_item(detail))
        time.sleep(REQUEST_DELAY)

        if i % 50 == 0:
            print(f"  ...progress: {i}/{len(all_item_ids)} item processed")

    print(f"getItem finished. {len(all_rows)} item successfully processed, {skipped} item skipped (delisted/error).")

    if all_rows:
        inserted = insert_listings(all_rows)
        print(f"[{datetime.now()}] Finished. {inserted} row successfully inserted to raw.ebay_listings.")
    else:
        print("No data to insert.")

if __name__ == "__main__":
    main()