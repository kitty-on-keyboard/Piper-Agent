from config import load_config


def test_good(tmp_path):
    p = tmp_path / "good.json"
    p.write_text('{"a": 1}')
    assert load_config(str(p)) == {"a": 1}


def test_malformed_returns_empty(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json at all")
    assert load_config(str(p)) == {}
