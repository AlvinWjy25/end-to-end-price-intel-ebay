with source as (
    select * from {{ source('raw', 'ebay_listings') }}
),

cleaned as (
    select
        item_id,
        fetched_at,
        case 
            when price > 0 then price 
            else null 
        end as price,
        trim(title) as title,
        upper(trim(currency)) as currency,
        initcap(trim(condition)) as condition,
        upper(trim(seller_location)) as seller_location,
        trim(description) as description,

        case
            when title ~* '(?:vol\.?|vols\.?|volume|volumes)\s*\d+\s*-\s*(?:vol\.?|vols\.?|volume|volumes)?\s*\d+' or title ~* '(complete set|full set)'
            then true
            else false
        end as is_boxset
    from source
),

Feature_Engineering as (
    select
        *,        
        (regexp_match(title, '(?:vol\.?|vols\.?|volume|volumes)\s*(\d+(?:\.\d+)?)', 'i'))[1]::float as volume_number,
        (regexp_match(title, '(?:vol\.?|vols\.?|volume|volumes)\s*(\d+)\s*-\s*(?:vol\.?|vols\.?|volume|volumes)?\s*(\d+)', 'i'))[2]::float as volume_number_end,

        case    
            when ((is_boxset is true
             or description ~* '(?:vol\.?|vols\.?|volume|volumes)\s*\d+\s*-\s*\d+') and description ~* '\d+\.\d+')
            then true
            else false
        end as boxset_side_story_edition_included,

        case    
            when title ~* 'vol. \d+\.\d+'
            and title !~* '(?:vol\.?,?|vols\.?,?|volume|volumes)\s*\d+\s*-\s*\d+'
            then true
            else false
        end as standalone_side_story_edition,

        case 
            when title ~* '(exclusive|bonus|special|platinum|collector|booklet|limited|fanbook|signed|Obi)'
            then true
            else false
        end as is_special_edition

    from cleaned
    where item_id is not null
),

deduplicated as (
    select distinct on (item_id) *
    from Feature_Engineering
    order by item_id, fetched_at desc
),

light_novel_only as(
    select *
    from deduplicated
    where title !~* 'manga'
        and title !~* 'figure'
        and title !~* 'artbook'
        and title !~* 'poster'
        and title !~* 'blanket'
        and title !~* 'acrylic'
        and title !~* 't-shirt'
        and title !~* 'cosplay'
        and title !~* 'keychain'
        and title !~* 'sticker'
        and title !~* 'wall scroll'
        and title !~* 'phone case'
        and title !~* 'figurine'
        and title !~* 'diorama'
        and title !~* 'nendoroid'
        and title !~* 'figma'
        and title !~* ' statue'
        and title !~* ' scale'
        and title !~* 'prize'
        and title !~* 'bandai'
    order by item_id, fetched_at desc
)

select * from light_novel_only