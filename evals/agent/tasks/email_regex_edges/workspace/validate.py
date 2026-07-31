import re

PATTERN = re.compile(r".+@.+")


def is_valid_email(text):
    return PATTERN.match(text) is not None
