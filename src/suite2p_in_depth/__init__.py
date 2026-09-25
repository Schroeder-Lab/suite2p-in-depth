"""Two-photon depth-motion preprocessing tools."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("suite2p-in-depth")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
