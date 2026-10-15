from egonets_wl.data.validation import inspect_file


def test_missing_file_is_reported():
    result = inspect_file("data/raw/does-not-exist")
    assert result["exists"] is False
