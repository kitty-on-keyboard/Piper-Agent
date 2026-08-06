from score import clamp_score


def test_clamp_low():
    assert clamp_score(-5) == 0


def test_clamp_high():
    assert clamp_score(140) == 100


def test_clamp_mid():
    assert clamp_score(42) == 42
