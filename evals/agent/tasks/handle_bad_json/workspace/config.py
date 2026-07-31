import json


def load_config(path):
    with open(path) as fh:
        return json.load(fh)
