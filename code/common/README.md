# Shared mathematical core

This folder is the only shared implementation used by all three backbones.

- `parameterization.py` maps velocity and noise predictions to/from the
  clean-state anchor.
- `spectral.py` performs the exact local radial/tangential projection and the
  complementary low/high frequency split.
- `guidance.py` implements the seven reported methods and keeps each baseline
  in its reference-defined execution domain: CFG, CFG-Rescale, and C2FG use
  native outputs; APG, FDG, RPTG, and TFDG use clean-state predictions.
- `experiment.py` freezes scientific settings and the full paired sample list
  in an immutable manifest.

Backbone-specific scheduler logic, model loading, latent shapes, and sampler
updates do not belong here.
