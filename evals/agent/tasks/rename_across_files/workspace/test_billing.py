from billing import calc_total
from checkout import charge
from report import summary

ITEMS = [{"price": 2, "qty": 3}, {"price": 5, "qty": 1}]


def test_total():
    assert calc_total(ITEMS) == 11


def test_charge():
    assert charge(ITEMS, 20) == 9


def test_summary():
    assert summary(ITEMS) == "total: 11"
