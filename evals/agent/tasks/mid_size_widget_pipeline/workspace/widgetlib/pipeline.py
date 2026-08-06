from .pricing import price_order


def quote_order(lines):
    priced = price_order(lines)
    return {
        "currency": "USD",
        "lines": list(lines),
        "merchandise_cents": priced["merchandise_cents"],
        "shipping_cents": priced["shipping_cents"],
        "total_cents": priced["total_cents"],
        "weight_kg": priced["weight_kg"],
    }
