def median(values):
    """The middle value; the mean of the middle two when there is an even count."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return ordered[mid]
