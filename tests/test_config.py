from pathlib import Path

from egonets_wl.config import load_config


def test_default_config_is_mapping():
    assert load_config(Path("configs/default.yaml"))["project"]["name"] == "egonets-wl-kernel"
