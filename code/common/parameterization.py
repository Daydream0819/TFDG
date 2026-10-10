"""Conversions between native model outputs and clean predictions."""

import torch


def _broadcast_scalar(value, reference):
    value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    while value.ndim < reference.ndim:
        value = value.unsqueeze(-1)
    return value


def flow_velocity_to_x0(z_t, velocity, t):
    """Return x0 for z_t=(1-t)x0+t*noise and velocity=noise-x0."""
    return z_t - _broadcast_scalar(t, z_t) * velocity


def x0_to_flow_velocity(z_t, x0, t, eps=1e-6):
    """Inverse of :func:`flow_velocity_to_x0`."""
    return (z_t - x0) / _broadcast_scalar(t, z_t).clamp_min(eps)


def epsilon_to_x0(x_t, epsilon, alpha, sigma, eps=1e-8):
    """Return x0 for x_t=alpha*x0+sigma*epsilon."""
    alpha = _broadcast_scalar(alpha, x_t)
    sigma = _broadcast_scalar(sigma, x_t)
    return (x_t - sigma * epsilon) / alpha.clamp_min(eps)


def x0_to_epsilon(x_t, x0, alpha, sigma, eps=1e-8):
    """Inverse of :func:`epsilon_to_x0`."""
    alpha = _broadcast_scalar(alpha, x_t)
    sigma = _broadcast_scalar(sigma, x_t)
    return (x_t - alpha * x0) / sigma.clamp_min(eps)
