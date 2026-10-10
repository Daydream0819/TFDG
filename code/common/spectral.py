"""Frequency-local radial/tangential decomposition used by every RPTG variant."""

import torch


def frequency_masks(reference, cutoff=0.25):
    """Return complementary low/high rFFT masks for a BCHW tensor."""
    if not 0 < cutoff <= 1:
        raise ValueError("cutoff must be in (0, 1]")
    height, width = reference.shape[-2:]
    fy = torch.fft.fftfreq(height, device=reference.device, dtype=torch.float32)
    fx = torch.fft.rfftfreq(width, device=reference.device, dtype=torch.float32)
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    low = (radius <= 0.5 * cutoff).to(torch.float32)
    return low.view(1, 1, height, width // 2 + 1), (1.0 - low).view(
        1, 1, height, width // 2 + 1
    )


def spectral_geometry(cond, uncond, cutoff=0.25, eps=1e-8):
    """Return the shared local spectral geometry for RPTG analyses and samplers."""
    if cond.shape != uncond.shape:
        raise ValueError("conditional and unconditional predictions must match")
    cond_spectrum = torch.fft.rfft2(cond.float(), norm="ortho")
    residual = torch.fft.rfft2((cond - uncond).float(), norm="ortho")
    magnitude = torch.abs(cond_spectrum)
    valid = magnitude > eps
    safe_magnitude = torch.where(valid, magnitude, torch.ones_like(magnitude))
    # Use an exact unit vector whenever the reference phase is defined.  The
    # previous ``magnitude + eps`` form slightly shortened every basis vector
    # and therefore made the nominal projection only approximately orthogonal.
    unit = (cond_spectrum / safe_magnitude) * valid.to(cond_spectrum.dtype)
    radial_coefficient = torch.real(residual * torch.conj(unit))
    radial = radial_coefficient * unit
    tangential = residual - radial
    low, high = frequency_masks(cond, cutoff)
    return cond_spectrum, radial, tangential, radial_coefficient, low, high


def decompose_residual(cond, uncond, cutoff=0.25):
    """Split a clean-prediction residual into four orthogonal spectral components.

    The radial/tangential directions are defined independently at every complex
    Fourier coefficient relative to the conditional spectrum.  The returned
    tensors remain in rFFT space for direct method-specific reconstruction.
    """
    cond_spectrum, radial, tangential, _, low, high = spectral_geometry(
        cond, uncond, cutoff
    )
    return cond_spectrum, {
        "low_radial": low * radial,
        "low_tangential": low * tangential,
        "high_radial": high * radial,
        "high_tangential": high * tangential,
    }
