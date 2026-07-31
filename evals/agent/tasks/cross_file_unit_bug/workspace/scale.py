def read_grams(raw):
    """Raw sensor counts -> grams. 4 counts per gram."""
    return raw / 4.0
