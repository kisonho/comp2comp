"""Compatibility imports for TotalSegmentator Python API."""

from importlib.util import find_spec

if find_spec("totalsegmentatorv2") is not None:
    from totalsegmentatorv2.python_api import totalsegmentator
else:
    from totalsegmentator.python_api import totalsegmentator
