# End-to-End eBay Light Novel Price Intelligence: Agent Context & Architecture Reference

This document provides a comprehensive breakdown of the core logic, database schemas, data transformation pipelines, and function-level documentation for all `.py` and `.sql` files in the repository. It is designed to serve as an authoritative reference for AI agents working on or extending this project.

---

## Directory Structure
```
project_1/
├── config/
|   ├──.env
|   ├──config_script.py
|   └──requirements.txt   
├── logs/
│   ├── dbt_logs     
│   └── pipeline_logs    
├── src/
│   ├── artifacts/
│   │   ├── evaluation/
|   │   │   ├── classification/
|   │   │   └── regression/
|   │   └── models
|   │   │   ├── classification/
|   │   │   └── regression/
│   │   └── preprocessed
|   │   │   ├── classification/
|   │   │   └── regression/ (X_train, dst)
│   ├── ingestion/
│   │   ├── backups/
│   │   └── ingest.py
│   ├── train.py
│   ├── pipeline.py
│   ├── evaluate.py
│   └── preprocessor.py
├── notebook/
|   ├── 01_EDA.ipynb (regression - Done)
│   └── 02_EDA.ipynb (classification - not started)
├── price_intel_dbt/
│   ├── analyses/
│   ├── .user.yml
│   ├── profiles.yml
│   ├── dbt_project.yml
│   ├── models/
│   │   ├── fact/ 
│   │   |   ├── fct_ebay_listings.sql
│   │   |   ├── _schema.yml
│   │   ├── intermediate/
│   │   |   ├── int_ebay_listing_risk_analysis.sql
│   │   |   ├── _schema.yml
│   │   └── staging/
│   │       ├── stg_ebay_listings.sql
│   │       └── _sources.yml
│   └── macros/
│        ├── generate_schema_name.sql
│        └── .gitkeep
```


## 1. System Overview & Architecture

The project is an end-to-end price intelligence and machine learning pipeline that collects, cleans, analyzes, and models prices of Light Novel listings on eBay.

```
                      +----------------------------------+
                      |         eBay Browse API          |
                      +----------------------------------+
                                       |
                               (OAuth 2.0 / REST)
                                       v
                      +----------------------------------+
                      |     src/ingestion/ingest.py      |
                      +----------------------------------+
                                       |
                         (psycopg2 INSERT & JSON Backup)
                                       v
                      +----------------------------------+
                      | PostgreSQL DB: raw.ebay_listings |
                      +----------------------------------+
                                       |
                                  (dbt Run)
                                       v
     +-------------------------------------------------------------------+
     |                         dbt Data Pipeline                         |
     |                                                                   |
     |  stg_ebay_listings  -->  int_ebay_listing_risk_analysis  -->  ... |
     |  (Clean & Flatten)        (Tiered Volume & Risk Rules)            |
     |                                                                   |
     |  ...  -->  fct_ebay_listings                                      |
     |            (Final Fact Table / Feature Store)                     |
     +-------------------------------------------------------------------+
                                       |
                               (SQLAlchemy / pandas)
                                       v
                      +----------------------------------+
                      |    config/config_script.py       |
                      |   (Database loader utilities)    |
                      +----------------------------------+
                                       |
                                       v
                      +----------------------------------+
                      |       src/preprocessor.py        |
                      |  (Encoders, Splits & Diagnostics)|
                      +----------------------------------+
                                       |
                                       v
                      +----------------------------------+
                      |          src/pipeline.py         |
                      |       (Pipeline Orchestration)   |
                      +----------------------------------+
                                       |
                                       v
                      +----------------------------------+
                      |     src/train.py & evaluate.py   |
                      |      (ML Model Training & Eval)  |
                      +----------------------------------+
```

---

## 2. Database Schemas & Data Models

### 2.1 Raw Table Schema: `raw.ebay_listings` (PostgreSQL)
Populated directly by [`src/ingestion/ingest.py`](file:///c:/Users/Alvin/Music/project_1/src/ingestion/ingest.py).

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `item_id` | `VARCHAR` (PK) | Unique eBay item ID (e.g. `v1\|307070835585\|0`). |
| `title` | `TEXT` | Raw listing title. |
| `price` | `NUMERIC` | Listed item price. |
| `currency` | `VARCHAR` | Currency code (e.g., `USD`). |
| `condition` | `TEXT` | Item condition (e.g., `Brand New`, `Like New`, `Good`). |
| `seller_location` | `TEXT` | Country of origin/seller location. |
| `description` | `TEXT` | Plain text cleaned seller description (stripped of HTML). |
| `localized_aspects`| `JSONB` | Array of eBay item specifics (`[{"name": "...", "value": "..."}, ...]`). |
| `created_at` | `TIMESTAMPTZ` | Timestamp when the record was ingested into PostgreSQL. |

---

### 2.2 Staging Layer: `stg_ebay_listings` (dbt View / Model)
Defined in [`price_intel_dbt/models/staging/stg_ebay_listings.sql`](file:///c:/Users/Alvin/Music/project_1/price_intel_dbt/models/staging/stg_ebay_listings.sql).

- **Purpose**: Cleans raw strings, extracts relevant JSONB aspect fields into top-level columns (`aspect_book_title`, `aspect_issue_number`, `aspect_unit_of_sale`), derives basic text metrics (`title_length`, `title_word_count`, `is_boxset`), deduplicates records by `item_id`, and filters out non-light-novel merchandise (manga, figures, artbooks, cosplay, keychains, etc.).

---

### 2.3 Intermediate Layer: `int_ebay_listing_risk_analysis` (dbt Model)
Defined in [`price_intel_dbt/models/intermediate/int_ebay_listing_risk_analysis.sql`](file:///c:/Users/Alvin/Music/project_1/price_intel_dbt/models/intermediate/int_ebay_listing_risk_analysis.sql).

- **Purpose**: Performs multi-tiered volume extraction, volume count calculations, risk scoring for potential bootleg/fake listings, and assigns confidence ratings.

#### Multi-Tier Volume Extraction Logic:
1. **Tier 1 (High Confidence - Aspect Issue Range)**: Extracts range from `aspect_issue_number` (e.g. `"1-26"`). Uses negative lookahead `(?!\s*(?:st|nd|rd|th))` to reject printing edition notations (e.g. `#6-1ST`).
2. **Tier 1b (High Confidence - Aspect Issue Single)**: Extracts single volume number from `aspect_issue_number` if it contains a standalone number (e.g. `"24"`).
3. **Tier 2 (High Confidence - Aspect Book Title)**: Parses volume number from `aspect_book_title` matching `Vol. N`.
4. **Tier 3 (Medium Confidence - Title Regex Range & Single)**: Searches title for volume ranges/numbers (`Vol N-M`, `Vol N`, `#N`, including `part/parts` aliases).
5. **Tier 4 (Low Confidence - Description Fallback)**: Scans product description when title lacks numerical digits.

#### Risk Scoring System:
- **Text Risk Score (+70)**: Triggers on keywords like `reprint`, `bootleg`, `custom print`, `reading edition`, `pdf/ebook`, `not original`, or missing condition. Features negation protection (e.g., `"not a reprint"` or `"no bootleg"` within 4 words prevents false positives).
- **Price Risk Score (+15 or +30)**: Triggers if boxset price per volume is `< $5.00` (+30) or single listing price is `< $10.00` with non-used/acceptable condition (+15).
- **Risk Category**: `High Risk` if `total_risk_score >= 50`, else `Low Risk`.

---

### 2.4 Mart Layer: `fct_ebay_listings` (dbt Fact Model / Table)
Defined in [`price_intel_dbt/models/marts/fct_ebay_listings.sql`](file:///c:/Users/Alvin/Music/project_1/price_intel_dbt/models/marts/fct_ebay_listings.sql).

- **Purpose**: Serves as the primary feature store for ML modeling and analytics.

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `item_id` | `VARCHAR` | Primary Key. |
| `title` | `TEXT` | Listing title. |
| `title_length` | `INTEGER` | Character count of title. |
| `title_word_count` | `INTEGER` | Word count of title. |
| `currency` | `VARCHAR` | Currency indicator. |
| `condition` | `TEXT` | Item condition string. |
| `seller_location` | `TEXT` | Seller location country code. |
| `is_boxset` | `BOOLEAN` | True if listing represents a multi-volume set/bundle. |
| `is_special_edition` | `BOOLEAN` | True if title indicates special/limited edition, signed copy, obi, etc. |
| `boxset_side_story_edition_included` | `BOOLEAN` | True if boxset includes side story / fractional volume (e.g. Vol 11.5). |
| `standalone_side_story_edition` | `BOOLEAN` | True if standalone volume is a side-story / fractional volume. |
| `price` | `NUMERIC` | Total listed item price. |
| `price_per_volume` | `NUMERIC` | Calculated unit price (`price / volume_count`). |
| `volume_number` | `FLOAT` | Extracted volume number (or start of range). |
| `volume_number_end` | `FLOAT` | Extracted end volume number (for sets). |
| `volume_count` | `INTEGER` | Derived total volumes included in listing. |
| `text_risk_score` | `INTEGER` | Anomaly risk score from text parsing. |
| `price_risk_score` | `INTEGER` | Anomaly risk score from price thresholding. |
| `total_risk_score` | `INTEGER` | Combined text and price risk score. |
| `risk_category` | `VARCHAR` | `'High Risk'` or `'Low Risk'`. |
| `volume_confidence` | `VARCHAR` | Confidence level of volume extraction (`'high'`, `'medium'`, `'low'`). |

---

## 3. Python Module & Function Summaries

### 3.1 [`config/config_script.py`](file:///c:/Users/Alvin/Music/project_1/config/config_script.py)
Configuration, dynamic logger setup, and database connector helper class.

- **`setup_logger(run_type: str) -> logging.Logger`**
  - Configures dual log handlers (timestamped file in `logs/pipeline_logs/` and clean console stream).
- **`class load_dataframe`**
  - **`__init__(self)`**: Initializes PostgreSQL database connection configuration variables from environment parameters or default fallbacks.
  - **`create_connection(self)`**: Builds a SQLAlchemy database engine (`postgresql://...`).
  - **`load_regression_data(self) -> pd.DataFrame`**: Queries `public.fct_ebay_listings` for regression dataset.
  - **`load_classification_data(self) -> pd.DataFrame`**: Queries `public.fct_ebay_listings` for classification dataset.
  - **`fit(self) -> tuple[pd.DataFrame, pd.DataFrame]`**: Executes engine setup and returns regression and classification dataframes.

---

### 3.2 [`src/ingestion/ingest.py`](file:///c:/Users/Alvin/Music/project_1/src/ingestion/ingest.py)
Primary data ingestion pipeline pulling listings from the eBay Browse API into PostgreSQL.

- **`get_access_token() -> str`**
  - Requests a 2-hour OAuth 2.0 token from eBay Identity API using `EBAY_CLIENT_ID` and `EBAY_CLIENT_SECRET` via Client Credentials Grant.
- **`search_item_ids(session: requests.Session, access_token: str, keyword: str, limit: int = 30) -> list[str]`**
  - Calls `item_summary/search` endpoint to retrieve matching item IDs for a query term.
- **`get_item_detail(session: requests.Session, access_token: str, item_id: str) -> dict | None`**
  - Calls `getItem` endpoint for full item metadata (including full raw HTML seller description and `localizedAspects`).
- **`strip_html(raw_html: str | None) -> str | None`**
  - Uses BeautifulSoup to parse raw description HTML and strip all tags, returning plain text.
- **`create_retry_session() -> requests.Session`**
  - Configures an HTTP session with exponential backoff retries for transient status codes (`429`, `500`, `502`, `503`, `504`).
- **`parse_item(item: dict) -> tuple`**
  - Extracts and formats key fields (`item_id`, `title`, `price`, `currency`, `condition`, `seller_location`, stripped `description`, and `localized_aspects` as `psycopg2.extras.Json`).
- **`insert_listings(rows: list[tuple]) -> int`**
  - Batch executes PostgreSQL `INSERT INTO raw.ebay_listings ... ON CONFLICT (item_id) DO UPDATE`.
- **`save_backup_to_json(rows: list[tuple], filename: str)`**
  - Writes ingested batch rows to a timestamped local JSON file in `backups/` prior to DB insertion.
- **`main()`**
  - Entry point executing multi-keyword search loop across light novel search terms and variants, deduplicating IDs, fetching full details, backing up, and persisting to Postgres.

---

### 3.3 [`src/preprocessor.py`](file:///c:/Users/Alvin/Music/project_1/src/preprocessor.py)
Data cleaning, feature encoding, train-test splitting, and pipeline diagnostic assertions.

- **`class preprocess_regression(load_dataframe)`**
  - **`__init__(self, random_state=42)`**: Inherits database loader, initializes logging and random seed.
  - **`ordinal_encoder(self, df: pd.DataFrame) -> pd.DataFrame`**:
    - Maps item `condition` strings to ordinal numeric scores (1 = `Acceptable` to 7 = `Brand New`).
    - Computes numeric `volume_tier_encoded` (1 = Single, 2 = Small Bundle [2-5], 3 = Medium Set [6-15], 4 = Large Set [16+]).
  - **`split_data(self, df_price_model: pd.DataFrame)`**:
    - Selects numeric, categorical, and boolean feature matrices.
    - Performs an 80/20 `train_test_split` on feature matrix `X` and target vector `y` (`price`).
    - Retains metadata columns (`item_id`, `title`, `is_special_edition`, `is_boxset`).
  - **`overview_dataframe(self, df: pd.DataFrame)`** *(staticmethod)*:
    - Diagnostic logger printing head, schema info, and summary statistics.
  - **`final_check(self)`**:
    - Diagnostic verification suite:
      - **Test 1**: Verifies no negative values exist in price target.
      - **Test 2**: Verifies no `NaN` values exist in feature matrix `X`.
      - **Test 3**: Verifies no negative values in `volume_count`.
  - **`fit_transform(self, df_raw: pd.DataFrame)`**:
    - Filters out high-risk listings (`risk_category != 'High Risk'`) and low-confidence volume parses (`volume_confidence != 'low'`).
    - Executes ordinal encoding, train-test splitting, diagnostic checks, and returns final datasets.

---

### 3.4 [`src/pipeline.py`](file:///c:/Users/Alvin/Music/project_1/src/pipeline.py)
- **Purpose**: Main execution orchestrator script. Loads data via `preprocess_regression().fit()` and runs full preprocessing via `fit_transform()`.

---

### 3.5 Diagnostic Scripts ([`src/ingestion/debug/`](file:///c:/Users/Alvin/Music/project_1/src/ingestion/debug))

- **[`src/ingestion/debug/check_unit_style.py`](file:///c:/Users/Alvin/Music/project_1/src/ingestion/debug/check_unit_style.py)**
  - `aspects_to_dict(localized_aspects)`: Flattens aspect array into dictionary.
  - `main()`: Inspects raw JSON responses in `debug_output/raw/` to analyze coverage of specific fields like `Unit of Sale`, `Style`, `Format`, `Genre`, `Intended Audience`.
- **[`src/ingestion/debug/ingest_v2.py`](file:///c:/Users/Alvin/Music/project_1/src/ingestion/debug/ingest_v2.py)**
  - Sampling probe script that fetches API responses for a small sample of keywords to evaluate coverage of `localizedAspects`, `Features`, `Issue Number`, and `Series Title` without inserting to Postgres.
- **[`src/ingestion/debug/inspect_ebay_response.py`](file:///c:/Users/Alvin/Music/project_1/src/ingestion/debug/inspect_ebay_response.py)**
  - Utility script dumping raw eBay `getItem` JSON responses to `debug_output/` for manual field structure inspection.

---

### 3.6 Placeholder Modules
- **[`src/train.py`](file:///c:/Users/Alvin/Music/project_1/src/train.py)**: Placeholder module for model training logic (XGBoost / LightGBM / Ridge regression).
- **[`src/evaluate.py`](file:///c:/Users/Alvin/Music/project_1/src/evaluate.py)**: Placeholder module for model evaluation metrics (RMSE, MAE, R²).

---

## 4. SQL Macro & Model Summaries

### 4.1 Macro: [`price_intel_dbt/macros/generate_schema_name.sql`](file:///c:/Users/Alvin/Music/project_1/price_intel_dbt/macros/generate_schema_name.sql)
- **`generate_schema_name(custom_schema_name, node)`**: Custom dbt macro that overrides dbt's default schema concatenation behavior, returning the custom schema name directly when provided.

---

### 4.2 Staging Model: [`price_intel_dbt/models/staging/stg_ebay_listings.sql`](file:///c:/Users/Alvin/Music/project_1/price_intel_dbt/models/staging/stg_ebay_listings.sql)
- **CTEs**:
  - `source`: Selects from `{{ source('raw', 'ebay_listings') }}`.
  - `cleaned`: Cleans price, title, currency, condition, seller location, and derives initial `is_boxset` boolean flag based on title range patterns.
  - `aspects_flattened`: Uses PostgreSQL `jsonb_array_elements()` to pull specific aspect keys (`Book Title`, `Issue Number`, `Unit of Sale`) into flat columns.
  - `simple_title_analysis`: Computes `title_length` and `title_word_count`.
  - `joined`: Merges cleaned listing attributes, flattened aspect attributes, and title metrics.
  - `deduplicated`: Evaluates `distinct on (item_id)` sorted by `fetched_at desc`.
  - `light_novel_only`: Excludes non-light-novel keywords (`manga`, `figure`, `artbook`, `cosplay`, `nendoroid`, `figma`, etc.).

---

### 4.3 Intermediate Model: [`price_intel_dbt/models/intermediate/int_ebay_listing_risk_analysis.sql`](file:///c:/Users/Alvin/Music/project_1/price_intel_dbt/models/intermediate/int_ebay_listing_risk_analysis.sql)
- **CTEs**:
  - `source_data`: References `{{ ref('stg_ebay_listings') }}`.
  - `volume_extraction`: Evaluates regular expression patterns for Tiers 1 through 4 (Tier 1 issue range, Tier 1b single issue, Tier 2 book title, Tier 3 title range/single/hash, Tier 4 description fallback), and flags special/bonus/side-story editions.
  - `volume_resolved`: Implements tier fallback precedence rules, resolves `volume_number` and `volume_number_end`, assigns `volume_confidence` (`high`, `medium`, `low`), and sets `title_desc_mismatch` QA flags.
  - `metrics_calculation`: Calculates `price_per_volume` (`price / volume_count`) and resolves `volume_count`.
  - `risk_scoring`: Computes `text_risk_score` (negation-aware regex searching for bootlegs/reprints) and `price_risk_score` (abnormally low unit/listing price).
  - `Final Select`: Sums `total_risk_score` and assigns `risk_category` (`High Risk` vs `Low Risk`).

---

### 4.4 Mart Model: [`price_intel_dbt/models/marts/fct_ebay_listings.sql`](file:///c:/Users/Alvin/Music/project_1/price_intel_dbt/models/marts/fct_ebay_listings.sql)
- **Query**: Selects cleaned attributes, unit pricing, extracted volume numbers, risk scores, risk categories, and confidence flags from `int_ebay_listing_risk_analysis`, forming the final dataset ready for analysis and ML ingestion.
