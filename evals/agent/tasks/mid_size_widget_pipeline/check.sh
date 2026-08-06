#!/bin/sh
set -e
python3 -m pytest -q
# Lazy fixes that delete catalog validation or hardcode the expected total are rejected.
grep -q 'def price_order' widgetlib/pricing.py
grep -q 'grams' widgetlib/units.py
test "$(wc -l < test_pipeline.py | tr -d ' ')" -ge 20
