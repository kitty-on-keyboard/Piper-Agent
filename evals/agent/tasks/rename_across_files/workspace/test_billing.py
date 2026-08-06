from checkout import charge
from invoices import invoice_lines
from report import summary

ITEMS = [{"price": 2, "qty": 3}, {"price": 5, "qty": 1}]


def test_charge():
    assert charge(ITEMS, 20) == 9


def test_summary():
    assert summary(ITEMS) == "total: 11"


def test_invoice_lines():
    lines = invoice_lines(ITEMS)
    assert lines[-1] == "due: 11"
    assert len(lines) == 3
