import pytest

from egonets_wl.data.download import download_file


def test_download_rejects_unverified_non_https_url(tmp_path):
    with pytest.raises(ValueError):
        download_file("http://example.test/archive.zip", tmp_path, tmp_path / "manifest.json")
