"""
SAVE acquisition policies.

Source: Spec §2.4; Blueprint §1.2 (acquisition module).
"""

from save.acquisition.base import AcquisitionPolicy
from save.acquisition.residual_variance import ResidualVarianceAcquisition
from save.acquisition.self_entropy import SelfEntropyAcquisition
from save.acquisition.surrogate_entropy import SurrogateEntropyAcquisition
from save.acquisition.uniform import UniformAcquisition

__all__ = [
    "AcquisitionPolicy",
    "ResidualVarianceAcquisition",
    "SelfEntropyAcquisition",
    "SurrogateEntropyAcquisition",
    "UniformAcquisition",
]
