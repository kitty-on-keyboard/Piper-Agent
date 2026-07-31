from shipping import shipping_cost


def test_two_kilos_at_three_per_kilo():
    # 8000 raw counts = 2000 g = 2 kg, at 3/kg = 6.
    assert shipping_cost(8000, 3) == 6
