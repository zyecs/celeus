"""
CELEUS acquisition policies.

Source: Spec §2.4; Blueprint §1.2 (acquisition module).
"""

from celeus.acquisition.base import AcquisitionPolicy
from celeus.acquisition.residual_variance import ResidualVarianceAcquisition
from celeus.acquisition.self_entropy import SelfEntropyAcquisition
from celeus.acquisition.surrogate_entropy import SurrogateEntropyAcquisition
from celeus.acquisition.uniform import UniformAcquisition

__all__ = [
    "AcquisitionPolicy",
    "ResidualVarianceAcquisition",
    "SelfEntropyAcquisition",
    "SurrogateEntropyAcquisition",
    "UniformAcquisition",
]
