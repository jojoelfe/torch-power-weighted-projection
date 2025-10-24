"""Create power weighted projections of volumes"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-power-weighted-projection")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Johannes Elferich"
__email__ = "jojotux123@hotmail.com"

from .projection import power_projection_raycast_gpu
from .utils import zyz_to_rotation_matrix

__all__ = [
    "power_projection_raycast_gpu",
    "zyz_to_rotation_matrix",
]
