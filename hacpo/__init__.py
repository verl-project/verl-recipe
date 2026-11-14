"""HACPO recipe for heterogeneous policy collaboration in verl."""

from .hacpo_core_algos import CapabilityTracker, HacpoLossConfig, compute_hacpo_advantages, compute_hacpo_loss
from .hacpo_trajectory import Message, TrainingView, Trajectory

__all__ = [
    "CapabilityTracker",
    "HacpoLossConfig",
    "Message",
    "TrainingView",
    "Trajectory",
    "compute_hacpo_advantages",
    "compute_hacpo_loss",
]
