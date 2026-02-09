"""Inference module."""
from .earley_parser import ProbabilisticEarleyParser, EarleyState

__all__ = [
    'ProbabilisticEarleyParser',
    'EarleyState',
]
