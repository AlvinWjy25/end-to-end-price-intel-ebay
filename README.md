docker run --name price-intel-db `
  -e POSTGRES_USER=alvin `
  -e POSTGRES_PASSWORD=devpassword `
  -e POSTGRES_DB=price_intelligence `
  -p 5432:5432 `
  -v pgdata:/var/lib/postgresql/data `
  -d postgres:16 

Powershell: docker run --name price-intel-db -e POSTGRES_USER=alvin -e POSTGRES_PASSWORD=devpassword -e POSTGRES_DB=price_intelligence -p 5432:5432 -v pgdata:/var/lib/postgresql/data -d postgres:16

dbt init price_intel_dbt
`config`:
    type: postgres
    host: localhost
    user: alvin
    password: devpassword
    port: 5432
    dbname: price_intelligence
    schema: dbt_dev
    
dbt debug