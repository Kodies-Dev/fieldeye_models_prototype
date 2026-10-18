"""stitchlive - skladanie zdjec z drona w czasie rzeczywistym na bazie OpenCV."""

__version__ = "0.1.0"

from .basic import BasicStitcher
from .detail import DetailStitcher
from .global_stitcher import GlobalStitcher
from .source import FolderSource, BatchSource

__all__ = ["BasicStitcher", "DetailStitcher", "GlobalStitcher",
           "FolderSource", "BatchSource"]
