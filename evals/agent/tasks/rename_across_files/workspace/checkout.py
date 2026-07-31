from billing import calc_total


def charge(items, balance):
    due = calc_total(items)
    return balance - due
