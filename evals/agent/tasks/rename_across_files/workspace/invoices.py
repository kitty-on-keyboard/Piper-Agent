from billing import calc_total


def invoice_lines(items):
    total = calc_total(items)
    return [f"{item['qty']} x {item['price']}" for item in items] + [f"due: {total}"]
