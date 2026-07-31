import store


def setup_function():
    store._CACHE.clear()
    store._ROWS.clear()


def test_reads_what_was_written():
    store.save_user(1, {"name": "ada"})
    assert store.get_user(1) == {"name": "ada"}


def test_sees_a_later_write():
    store.save_user(1, {"name": "ada"})
    assert store.get_user(1) == {"name": "ada"}
    store.save_user(1, {"name": "grace"})
    assert store.get_user(1) == {"name": "grace"}


def test_missing_user_is_empty():
    assert store.get_user(99) == {}
