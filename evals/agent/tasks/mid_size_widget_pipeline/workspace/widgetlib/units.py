def grams_to_kg(grams):
    return grams / 1000.0


def shipping_weight_kg(items):
    """items: iterable of (grams, qty)."""
    total_grams = sum(grams * qty for grams, qty in items)
    # Bug: forgets to convert to kilograms, so shipping thinks grams are kg.
    return total_grams
