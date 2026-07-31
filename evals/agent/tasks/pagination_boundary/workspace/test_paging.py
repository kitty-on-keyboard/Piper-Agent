from paging import page_slice

ROWS = list(range(1, 11))  # 1..10


def test_first_page():
    assert page_slice(ROWS, 1, 3) == [1, 2, 3]


def test_second_page():
    assert page_slice(ROWS, 2, 3) == [4, 5, 6]


def test_last_partial_page():
    assert page_slice(ROWS, 4, 3) == [10]


def test_page_past_the_end():
    assert page_slice(ROWS, 9, 3) == []


def test_per_page_larger_than_rows():
    assert page_slice(ROWS, 1, 50) == ROWS
