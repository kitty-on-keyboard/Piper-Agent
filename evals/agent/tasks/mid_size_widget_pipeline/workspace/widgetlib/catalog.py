CATALOG = {
    "bolt": {"sku": "bolt", "unit_price_cents": 25, "weight_grams": 12},
    "washer": {"sku": "washer", "unit_price_cents": 10, "weight_grams": 4},
    "plate": {"sku": "plate", "unit_price_cents": 350, "weight_grams": 800},
}


def lookup(sku):
    item = CATALOG.get(sku)
    if item is None:
        raise KeyError(sku)
    return dict(item)
