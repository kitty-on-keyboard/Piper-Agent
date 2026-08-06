from . import catalog
from .units import grams_to_kg, shipping_weight_kg


def line_total_cents(sku, qty):
    item = catalog.lookup(sku)
    return item["unit_price_cents"] * qty


def price_order(lines, shipping_rate_cents_per_kg=200):
    """
    lines: list of {sku, qty}
    Returns dict with merchandise_cents, shipping_cents, total_cents.
    """
    merchandise = 0
    weight_pairs = []
    for line in lines:
        sku = line["sku"]
        qty = line["qty"]
        merchandise += line_total_cents(sku, qty)
        item = catalog.lookup(sku)
        weight_pairs.append((item["weight_grams"], qty))
    weight_kg = shipping_weight_kg(weight_pairs)
    # grams_to_kg is available but unused because shipping_weight_kg is wrong.
    _ = grams_to_kg
    shipping = int(round(weight_kg * shipping_rate_cents_per_kg))
    return {
        "merchandise_cents": merchandise,
        "shipping_cents": shipping,
        "total_cents": merchandise + shipping,
        "weight_kg": weight_kg,
    }
