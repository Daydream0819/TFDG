"""Seven reported guidance methods plus one calibration-only policy.

CFG, CFG-Rescale, and C2FG operate on the native network prediction. APG and
FDG are defined on clean denoiser predictions, as are RPTG and TFDG. The
dispatcher performs the fixed parameterization conversion only for this clean
group, which prevents a baseline from being silently redefined.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

METHODS = (
    "cfg",
    "cfg_rescale",
    "c2fg",
    "apg",
    "fdg",
    "rptg",
    "rptg_high",
    "tfdg",
)

# Methods reported in the paper. RPTG variants remain implementation-level
# calibration diagnostics and must not silently enter a comparison table.
REPORTED_METHODS = ("cfg", "cfg_rescale", "apg", "fdg", "c2fg", "tfdg")

NATIVE_METHODS = frozenset({"cfg", "cfg_rescale", "c2fg"})
# ``rptg_high`` is a calibration-only intermediate policy.  It is deliberately
# kept separate from the six reported non-TFDG baselines.
CLEAN_METHODS = frozenset({"apg", "fdg", "rptg", "rptg_high", "tfdg"})


@dataclass
class GuidanceConfig:
    method: str = "cfg"
    scale: float = 1.0
    momentum: float = -0.5
    norm_threshold: float = 0.0
    parallel_weight: float = 0.0
    fdg_low_scale: float = 1.0
    cutoff: float = 0.25
    rescale_factor: float = 0.7
    c2fg_lambda: float = 1.0
    low_min_cfg: float = 1.0
    # ``None`` retains the original schedule and recovers to the global CFG
    # scale. A finite ceiling supports the partial-recovery policy selected on
    # a backbone-specific development split.
    low_max_cfg: float | None = None
    high_min_cfg: float = 3.0
    high_max_cfg: float | None = None
    low_eta: float = 0.5
    high_eta: float = 1.0

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"Unknown method {self.method!r}; choose from {METHODS}")
        if self.scale < 1:
            raise ValueError("scale must be >= 1")
        if self.norm_threshold < 0:
            raise ValueError("norm_threshold must be non-negative")
        if self.fdg_low_scale < 1:
            raise ValueError("FDG low-frequency guidance scale must be >= 1")
        if not 0 < self.cutoff <= 1:
            raise ValueError("cutoff must be in (0, 1]")
        if self.rescale_factor < 0 or self.rescale_factor > 1:
            raise ValueError("rescale_factor must be in [0, 1]")
        if self.c2fg_lambda < 0:
            raise ValueError("c2fg_lambda must be non-negative")
        if self.low_min_cfg < 1 or self.high_min_cfg < 1:
            raise ValueError("TFDG minimum effective CFG weights must be >= 1")
        if self.method in {"rptg_high", "tfdg"} and (
            self.low_min_cfg > self.scale or self.high_min_cfg > self.scale
        ):
            raise ValueError("TFDG minimum weights cannot exceed the final CFG scale")
        if self.low_max_cfg is not None:
            if not 1 <= self.low_max_cfg <= self.scale:
                raise ValueError("low_max_cfg must lie in [1, scale]")
            if self.low_min_cfg > self.low_max_cfg:
                raise ValueError("low_min_cfg cannot exceed low_max_cfg")
        if self.high_max_cfg is not None:
            if not 1 <= self.high_max_cfg <= self.scale:
                raise ValueError("high_max_cfg must lie in [1, scale]")
            if self.high_min_cfg > self.high_max_cfg:
                raise ValueError("high_min_cfg cannot exceed high_max_cfg")
        if self.low_eta < 0 or self.high_eta < 0:
            raise ValueError("TFDG eta values must be non-negative")


@dataclass
class GuidanceState:
    momentum_buffer: torch.Tensor | None = None

    def reset(self):
        self.momentum_buffer = None


@dataclass(frozen=True)
class GuidanceContext:
    """Physical schedule values needed by temporal guidance rules."""

    alpha: float = 1.0
    sigma: float = 1.0
    noise_time: float = 0.5
    step_index: int = 0
    total_steps: int = 1


def _dims(x):
    return tuple(range(1, x.ndim))


def _project(update, reference):
    dtype = update.dtype
    update64 = update.double()
    reference64 = F.normalize(reference.double(), dim=_dims(reference))
    parallel = (update64 * reference64).sum(
        dim=_dims(update64), keepdim=True
    ) * reference64
    return parallel.to(dtype), (update64 - parallel).to(dtype)


def _momentum(update, state, beta, update_state):
    previous = state.momentum_buffer
    if previous is None:
        previous = torch.zeros_like(update)
    running = update + beta * previous
    if update_state:
        state.momentum_buffer = running.detach()
    return running


def _clip_absolute(update, threshold):
    if threshold <= 0:
        return update
    norm = torch.linalg.vector_norm(update.float(), dim=_dims(update), keepdim=True)
    factor = torch.clamp(threshold / (norm + 1e-12), max=1.0)
    return update * factor.to(update.dtype)


def _apg(cond, uncond, cfg, state, update_state):
    """APG in clean-prediction space.

    This follows the original conditional formulation: normalize the guidance
    residual, suppress its parallel component, and extrapolate from the
    conditional prediction with ``(w - 1)``.
    """
    diff = _momentum(cond - uncond, state, cfg.momentum, update_state)
    diff = _clip_absolute(diff, cfg.norm_threshold)
    parallel, orthogonal = _project(diff, cond)
    return cond + (cfg.scale - 1.0) * (orthogonal + cfg.parallel_weight * parallel)


def _laplacian_two_levels(x):
    try:
        from kornia.geometry.transform import build_laplacian_pyramid

        return list(build_laplacian_pyramid(x, 2))
    except ImportError as error:
        raise RuntimeError(
            "FDG requires kornia==0.7.4 to reproduce the published "
            "Laplacian-pyramid implementation"
        ) from error


def _reconstruct_two_levels(pyramid):
    high, low = pyramid
    from kornia.geometry.transform import pyrup

    low = pyrup(low, border_type="reflect", align_corners=False)[
        :, :, : high.shape[-2], : high.shape[-1]
    ]
    return high + low


def _fdg(cond, uncond, cfg):
    """FDG in clean-prediction space using a two-level Laplacian pyramid."""
    cond_pyramid = _laplacian_two_levels(cond.float())
    uncond_pyramid = _laplacian_two_levels(uncond.float())
    guided = [
        c + (scale - 1.0) * (c - u)
        for c, u, scale in zip(
            cond_pyramid, uncond_pyramid, (cfg.scale, cfg.fdg_low_scale)
        )
    ]
    return _reconstruct_two_levels(guided).to(cond.dtype)


def _rptg_update(cond, uncond, cfg):
    """Pure RPTG: tangential residual at every frequency."""
    from .spectral import decompose_residual

    _, components = decompose_residual(cond, uncond, cutoff=cfg.cutoff)
    tangential = components["low_tangential"] + components["high_tangential"]
    return torch.fft.irfft2(tangential, s=cond.shape[-2:], norm="ortho").to(cond.dtype)


def _rptg(cond, uncond, cfg):
    selected = _rptg_update(cond, uncond, cfg)
    return cond + (cfg.scale - 1.0) * selected


def _rptg_high(cond, uncond, cfg, context):
    """Calibration-only RPTG with a scheduled high-frequency radial branch.

    The low-frequency radial residual remains disabled.  This isolated policy
    tests whether restoring detail-related radial guidance is sufficient to
    recover the quality lost by pure tangential RPTG.
    """
    from .spectral import decompose_residual

    _, components = decompose_residual(cond, uncond, cutoff=cfg.cutoff)
    high_ceiling = cfg.scale if cfg.high_max_cfg is None else cfg.high_max_cfg
    high_weight = _temporal_weight(
        cfg.high_min_cfg, cfg.high_eta, high_ceiling, context
    )
    tangential = components["low_tangential"] + components["high_tangential"]
    spectrum_update = (
        (cfg.scale - 1.0) * tangential
        + (high_weight - 1.0) * components["high_radial"]
    )
    update = torch.fft.irfft2(
        spectrum_update, s=cond.shape[-2:], norm="ortho"
    )
    return (cond + update).to(cond.dtype)


def _cfg_rescale(cond, uncond, cfg):
    """The standard CFG-Rescale correction on native model outputs."""
    guided = uncond + cfg.scale * (cond - uncond)
    dims = _dims(guided)
    guided_std = guided.float().std(dim=dims, keepdim=True)
    cond_std = cond.float().std(dim=dims, keepdim=True)
    rescaled = guided * cond_std / (guided_std + 1e-6)
    return (cfg.rescale_factor * rescaled + (1.0 - cfg.rescale_factor) * guided).to(
        cond.dtype
    )


def _effective_time(context):
    if context is None:
        return 0.5, 1.0, 1.0
    t = float(getattr(context, "noise_time", 0.5))
    alpha = float(getattr(context, "alpha", 1.0))
    sigma = float(getattr(context, "sigma", 1.0))
    return min(max(t, 0.0), 1.0), alpha, sigma


def _c2fg(cond, uncond, cfg, context):
    t, _, _ = _effective_time(context)
    # C2FG defines omega(t)=omega_0 exp(lambda(1-t/t_max)).  We report runs
    # using their shared *maximum* CFG weight at the clean endpoint, denoted
    # by cfg.scale.  Setting omega_0=cfg.scale*exp(-lambda) gives the exactly
    # equivalent normalized-clock form below, with t/t_max in [0, 1].
    effective = cfg.scale * math.exp(-cfg.c2fg_lambda * t)
    return uncond + effective * (cond - uncond)


def _temporal_weight(min_cfg, eta, ceiling_cfg, context):
    _, alpha, sigma = _effective_time(context)
    if eta == 0:
        g = 1.0
    else:
        # Stable form of [1 + eta*(sigma/alpha)^2]^-1/2. Unlike a clipped
        # ratio, this gives the exact continuous endpoint g=0 when alpha=0.
        denominator = math.sqrt(alpha * alpha + eta * sigma * sigma)
        g = abs(alpha) / max(denominator, 1e-12)
    return min_cfg + (ceiling_cfg - min_cfg) * g


def _tfdg(cond, uncond, cfg, context):
    from .spectral import decompose_residual

    _, components = decompose_residual(cond, uncond, cutoff=cfg.cutoff)
    low_ceiling = cfg.scale if cfg.low_max_cfg is None else cfg.low_max_cfg
    high_ceiling = cfg.scale if cfg.high_max_cfg is None else cfg.high_max_cfg
    low = _temporal_weight(cfg.low_min_cfg, cfg.low_eta, low_ceiling, context)
    high = _temporal_weight(cfg.high_min_cfg, cfg.high_eta, high_ceiling, context)
    tangential = cfg.scale
    guided_spectrum = (
        (low - 1.0) * components["low_radial"]
        + (high - 1.0) * components["high_radial"]
        + (tangential - 1.0) * components["low_tangential"]
        + (tangential - 1.0) * components["high_tangential"]
    )
    update = torch.fft.irfft2(guided_spectrum, s=cond.shape[-2:], norm="ortho")
    return (cond + update).to(cond.dtype)


def guidance_domain(method):
    """Return the prediction domain required by a guidance method."""
    if method in NATIVE_METHODS:
        return "native"
    if method in CLEAN_METHODS:
        return "clean"
    raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")


def _validate_pair(cond, uncond):
    if cond.shape != uncond.shape:
        raise ValueError("conditional and unconditional predictions must match")


def apply_native_guidance(
    cond,
    uncond,
    config,
    state=None,
    update_state=True,
    context=None,
):
    """Apply a method defined on the model's native prediction output."""
    _validate_pair(cond, uncond)
    if guidance_domain(config.method) != "native":
        raise ValueError(f"{config.method} requires native model predictions")
    # C2FG remains time-varying even when its clean-endpoint weight is one.
    if config.scale == 1 and config.method != "c2fg":
        return cond
    if config.method == "cfg":
        return uncond + config.scale * (cond - uncond)
    if config.method == "cfg_rescale":
        return _cfg_rescale(cond, uncond, config)
    if config.method == "c2fg":
        return _c2fg(cond, uncond, config, context)
    raise AssertionError(config.method)


def apply_clean_guidance(
    cond,
    uncond,
    config,
    state=None,
    update_state=True,
    context=None,
):
    """Apply a method defined on conditional/unconditional clean predictions."""
    _validate_pair(cond, uncond)
    if guidance_domain(config.method) != "clean":
        raise ValueError(f"{config.method} requires clean-state predictions")
    if state is None:
        state = GuidanceState()
    if config.scale == 1:
        return cond
    if config.method == "apg":
        return _apg(cond, uncond, config, state, update_state)
    if config.method == "fdg":
        return _fdg(cond, uncond, config)
    if config.method == "rptg":
        return _rptg(cond, uncond, config)
    if config.method == "rptg_high":
        return _rptg_high(cond, uncond, config, context)
    if config.method == "tfdg":
        return _tfdg(cond, uncond, config, context)
    raise AssertionError(config.method)


def apply_model_guidance(
    native_cond,
    native_uncond,
    config,
    *,
    to_clean,
    from_clean,
    state=None,
    update_state=True,
    context=None,
):
    """Dispatch guidance in its reference-defined prediction domain.

    ``to_clean`` converts one native prediction to a clean-state prediction;
    ``from_clean`` converts the guided clean prediction back to the model's
    native output.  They are not evaluated for native-domain methods.
    """
    if guidance_domain(config.method) == "native":
        return apply_native_guidance(
            native_cond,
            native_uncond,
            config,
            state=state,
            update_state=update_state,
            context=context,
        )
    clean_cond = to_clean(native_cond)
    clean_uncond = to_clean(native_uncond)
    clean_guided = apply_clean_guidance(
        clean_cond,
        clean_uncond,
        config,
        state=state,
        update_state=update_state,
        context=context,
    )
    return from_clean(clean_guided)


def apply_guidance(cond, uncond, config, state=None, update_state=True, context=None):
    """Backward-compatible domain-specific primitive.

    Callers must already provide tensors in :func:`guidance_domain`.  New model
    adapters should use :func:`apply_model_guidance` instead.
    """
    if guidance_domain(config.method) == "native":
        return apply_native_guidance(
            cond,
            uncond,
            config,
            state=state,
            update_state=update_state,
            context=context,
        )
    return apply_clean_guidance(
        cond,
        uncond,
        config,
        state=state,
        update_state=update_state,
        context=context,
    )
