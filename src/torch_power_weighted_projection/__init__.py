"""Create power weighted projections of volumes"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-power-weighted-projection")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Johannes Elferich"
__email__ = "jojotux123@hotmail.com"
