"""Shared, model-agnostic TFDG primitives.

Model-specific samplers live in ``sd35/``, ``dit/``, and ``edm2/``.  This
package contains only the mathematics that must remain identical across
prediction parameterizations.
"""

from .guidance import (
    CLEAN_METHODS,
    METHODS,
    NATIVE_METHODS,
    REPORTED_METHODS,
    GuidanceConfig,
    GuidanceContext,
    GuidanceState,
    apply_clean_guidance,
    apply_guidance,
    apply_model_guidance,
    apply_native_guidance,
    guidance_domain,
)
from .parameterization import (
    epsilon_to_x0,
    flow_velocity_to_x0,
    x0_to_epsilon,
    x0_to_flow_velocity,
)
from .spectral import decompose_residual, frequency_masks, spectral_geometry

__all__ = [
    "CLEAN_METHODS",
    "METHODS",
    "NATIVE_METHODS",
    "REPORTED_METHODS",
    "GuidanceConfig",
    "GuidanceContext",
    "GuidanceState",
    "apply_clean_guidance",
    "apply_guidance",
    "apply_model_guidance",
    "apply_native_guidance",
    "decompose_residual",
    "epsilon_to_x0",
    "flow_velocity_to_x0",
    "frequency_masks",
    "guidance_domain",
    "spectral_geometry",
    "x0_to_epsilon",
    "x0_to_flow_velocity",
]
