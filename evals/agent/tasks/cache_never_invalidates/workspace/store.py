_CACHE = {}
_ROWS = {}


def get_user(user_id):
    if user_id not in _CACHE:
        _CACHE[user_id] = dict(_ROWS.get(user_id, {}))
    return _CACHE[user_id]


def save_user(user_id, record):
    _ROWS[user_id] = dict(record)
