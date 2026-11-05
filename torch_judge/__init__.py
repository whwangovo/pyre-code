"""Pyre Code task definitions and judges.

The runnable entry points are the FastAPI grading service (``grading_service``) and the
Hot 100 judge (``torch_judge.algo``).
"""

from torch_judge._version import __version__

__all__ = ["__version__"]
