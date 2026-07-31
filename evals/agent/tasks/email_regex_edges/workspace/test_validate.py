import pytest

from validate import is_valid_email

GOOD = ["a@b.co", "first.last@example.com", "x+tag@sub.example.org"]
BAD = ["no-at-sign", "a@b", "@example.com", "a@@b.co", "a b@example.com",
       "trailing@example.com.", "a@.com"]


@pytest.mark.parametrize("addr", GOOD)
def test_accepts(addr):
    assert is_valid_email(addr)


@pytest.mark.parametrize("addr", BAD)
def test_rejects(addr):
    assert not is_valid_email(addr)
