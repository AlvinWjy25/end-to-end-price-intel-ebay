with source as (
    select * from {{ source('raw', 'ebay_listings') }}
),

cleaned as (
    select
        item_id,
        trim(title) as title,
        
        case 
            when price > 0 then price 
            else null 
        end as price,
        
        upper(trim(currency)) as currency,
        initcap(trim(condition)) as condition,
        upper(trim(seller_location)) as seller_location,
        trim(description) as description,
        fetched_at,

        -- Deteksi range volume dulu (Vol. 1-28, Vol 1-5, Vol.1-43, dst)
        -- Ini prioritas pertama karena kalau range terdeteksi, single volume_number jadi kurang bermakna
        case
            when title ~* '(?:vol\.?|vols\.?|volume)\s*\d+\s*-\s*\d+'
            then true
            else false
        end as is_boxset,

        (regexp_match(title, '(?:vol\.?|vols\.?|volume)\s*(\d+(?:\.\d+)?)', 'i'))[1]::float as volume_number,
        (regexp_match(title, '(?:vol\.?|vols\.?|volume)\s*(\d+)\s*-\s*(\d+)', 'i'))[2]::float as volume_number_end,

        case    
            when title ~* '(?:vol\.?|vols\.?|volume)\s*\d+\s*-\s*\d+'
             and title ~* '\d+\.\d+'
            then true
            else false
        end as boxset_special_edition_included,

        case    
            when title ~* 'vol. \d+\.\d+'
            and title !~* '(?:vol\.?,?|vols\.?,?|volume)\s*\d+\s*-\s*\d+'
            then true
            else false
        end as standalone_special_set

    from source
    where item_id is not null
),

deduplicated as (
    select distinct on (item_id) *
    from cleaned
    order by item_id, fetched_at desc
)

select * from deduplicated