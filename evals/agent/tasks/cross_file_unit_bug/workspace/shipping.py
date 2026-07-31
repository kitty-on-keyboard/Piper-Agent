from scale import read_grams


def shipping_cost(raw_counts, rate_per_kg):
    """Cost to ship a parcel, at `rate_per_kg` currency units per kilogram."""
    weight = read_grams(raw_counts)
    return weight * rate_per_kg
