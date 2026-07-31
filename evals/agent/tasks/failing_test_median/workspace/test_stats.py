from stats import median


def test_odd_count():
    assert median([3, 1, 2]) == 2


def test_even_count():
    assert median([4, 1, 3, 2]) == 2.5
