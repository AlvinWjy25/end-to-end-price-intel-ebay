-- models/marts/fct_ebay_listings.sql

with source as (
    select * from {{ ref('int_ebay_listing_risk_analysis') }}
)

select
    item_id,
    title,
    price,
    currency,
    condition,
    seller_location,
    volume_number,
    volume_number_end,
    is_boxset,
    boxset_side_story_edition_included,
    standalone_side_story_edition,
    is_special_edition,
    total_risk_score,
    risk_category,
    fetched_at

    -- description intentionally removed from final marts
    -- already used to account for risk_score in intermediate
    -- not required for regression/classification
    -- we can pull description data by matching the item_id from intermediate view table.

from source