#!/bin/bash

# Hentikan eksekusi jika ada perintah yang error
set -e

echo "Running DBT..."
cd price_intel_dbt
dbt run --profiles-dir . --log-path ../logs/dbt_logs
dbt test

echo "Running Python Pipeline..."
cd ../src
python pipeline.py