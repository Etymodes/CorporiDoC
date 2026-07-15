"""CorporiDoC research application."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("corporidoc")
except PackageNotFoundError:
    __version__ = "0+unknown"
