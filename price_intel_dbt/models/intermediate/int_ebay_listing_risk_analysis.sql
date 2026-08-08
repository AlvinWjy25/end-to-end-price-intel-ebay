with source_data as (
    select * from {{ ref('stg_ebay_listings') }}
),

risk_scoring as (
    select 
        *,
        -- Text risk score with negation-aware matching
        case
            when (
                title ~* '(reprint|unbranded|bootleg|pdf|ebook|custom print|reading edition)'
                or description ~* '(reprint|not original|loose|print on demand|not an official|not official|not suitable|fan translated|fan-translated|not from original|reading edition)'
            )
            and not (
                -- True Positive protection: negate the risk keywords within 4 words
                description ~* '(not|no|never|isn''t|isnt|without)\s+(\w+\s+){0,4}(reprint|unofficial|fan.?translated|bootleg)'
                or
                title ~* '(not|no|never)\s+(\w+\s+){0,4}(reprint|unofficial|bootleg)'
            )
            then 70
            else 0
        end as text_risk_score,

        case
            when is_boxset is true 
                and volume_number_end is not null
                and (price / NULLIF(volume_number_end - volume_number + 1, 0)) < 5.00  -- contoh threshold
            then 30

            when price < 10.00 and condition !~* 'acceptable|used' -- Allow 'Acceptable & used' condition below $8
            then 15

            else 0
        end as price_risk_score

    from source_data
)

select
    *,
    (text_risk_score + price_risk_score) as total_risk_score,

    case
        when (text_risk_score + price_risk_score) >= 50
            then 'High Risk'
        else 'Low Risk'
    end as risk_category

from risk_scoring