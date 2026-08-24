-- models/marts/fct_ebay_listings.sql

with source as (
    select * from {{ ref('int_ebay_listing_risk_analysis') }}
)

select
    item_id,
    title,
    title_length,
    title_word_count,
    currency,
    condition,
    seller_location,
    is_boxset,
    is_special_edition,
    is_first_print,
    total_bonus_count,
    has_signature,
    has_merch,
    has_paper_extra,
    boxset_side_story_edition_included,
    standalone_side_story_edition,
    price,
    price_per_volume,
    volume_number,
    volume_number_end,
    volume_count,
    text_risk_score,
    price_risk_score
    total_risk_score,
    is_ambiguous_bulk_pricing,
    risk_category,
    volume_confidence
from source