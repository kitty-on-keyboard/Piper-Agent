def page_slice(rows, page, per_page):
    """Rows for `page` (1-indexed), `per_page` at a time.

    A page past the end is empty rather than an error.
    """
    start = page * per_page
    return rows[start:start + per_page]
