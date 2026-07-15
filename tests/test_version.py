from importlib.metadata import version

import corporidoc


def test_runtime_version_matches_package_metadata() -> None:
    assert corporidoc.__version__ == version("corporidoc")
