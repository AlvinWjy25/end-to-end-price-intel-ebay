#!/bin/bash

# Stop execution if there is an error
set -e

# WARNING: This may consume your EBAY API Quota, SETUP YOUR EBAY API at https://developer.ebay.com/signin & Place as such:
# DIR: (parent dir)/config/.env
# .env content:  #DO NOT FORGET TO PUT config/.env on .GITIGNORE!

# EBAY_CLIENT_ID=your_ebay_client_id
# EBAY_CLIENT_SECRET=your_ebay_client_secret

# DB_USER=your_db_user
# DB_PASSWORD=your_db_password
# DB_HOST=your_db_host
# DB_PORT=your_db_port
# DB_NAME=your_db_name

# Run DBT
echo "Running DBT..."
cd price_intel_dbt
dbt run --profiles-dir . --log-path ../logs/dbt_logs
dbt test

# Run Python Pipeline
echo "Running Python Pipeline..."
cd ../src
python pipeline.py