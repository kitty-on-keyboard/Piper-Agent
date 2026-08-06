from widgetlib import quote_order
from widgetlib.units import grams_to_kg


def test_grams_to_kg():
    assert grams_to_kg(1000) == 1.0


def test_single_bolt_quote():
    quote = quote_order([{"sku": "bolt", "qty": 10}])
    # 10 * 25c merchandise; weight 120g = 0.12kg; shipping 0.12 * 200 = 24c
    assert quote["merchandise_cents"] == 250
    assert quote["weight_kg"] == 0.12
    assert quote["shipping_cents"] == 24
    assert quote["total_cents"] == 274


def test_mixed_order():
    quote = quote_order([
        {"sku": "washer", "qty": 50},
        {"sku": "plate", "qty": 1},
    ])
    # washer: 500c + 200g; plate: 350c + 800g; total weight 1.0 kg
    assert quote["merchandise_cents"] == 850
    assert abs(quote["weight_kg"] - 1.0) < 1e-9
    assert quote["shipping_cents"] == 200
    assert quote["total_cents"] == 1050


def test_unknown_sku_raises():
    try:
        quote_order([{"sku": "nope", "qty": 1}])
    except KeyError:
        return
    raise AssertionError("expected KeyError")
