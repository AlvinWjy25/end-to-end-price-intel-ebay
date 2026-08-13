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
    risk_category,
    volume_confidence
from source