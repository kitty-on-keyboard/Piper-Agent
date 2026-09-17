def clamp_score(value):
    """Clamps a numeric score into the range [0, 100]."""
    return max(0, min(100, value))
