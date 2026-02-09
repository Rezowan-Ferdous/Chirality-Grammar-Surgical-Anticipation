"""Training module."""
from .losses import (
    ActionLoss,
    DurationLoss,
    ObjectLoss,
    GoalLoss,
    GrammarKLLoss,
    PTGLoss
)

__all__ = [
    'ActionLoss',
    'DurationLoss',
    'ObjectLoss',
    'GoalLoss',
    'GrammarKLLoss',
    'PTGLoss',
]
