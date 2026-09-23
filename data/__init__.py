"""Training datasets."""

from .loader import load
from .processed_tcia import ProcessedTCIA
from .tcia import TCIA

__all__ = ["ProcessedTCIA", "TCIA"]
