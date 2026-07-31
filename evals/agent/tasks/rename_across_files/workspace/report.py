from billing import calc_total


def summary(items):
    return f"total: {calc_total(items)}"
