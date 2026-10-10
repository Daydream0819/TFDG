# TFDG: Time-Frequency Directional Guidance

Official implementation of **TFDG: Eliminating Visual Degradation in Image
Generation via Time-Frequency Directional Guidance**.

TFDG is a training-free classifier-free guidance method. It maps each
backbone's prediction into a common clean-sample estimate space, decomposes
the guidance residual into radial, tangential, low-frequency, and
high-frequency components, and assigns time-varying component weights during
reverse sampling.

## Supported backbones

| Backbone | Native prediction | Sampler | Code |
| --- | --- | --- | --- |
| Stable Diffusion v1.5 | Noise | DPM-Solver++ | `code/sd15/` |
| Stable Diffusion 3.5 Large | Velocity | Euler | `code/sd35/` |
| DiT-XL/2 | Noise | DDPM | `code/dit/` |
| EDM2-S | Direct clean prediction | EDM2 sampler | `code/edm2/` |

The four model folders contain only the adapters and download scripts. Model
weights, external repositories, generated images, evaluation caches, and
private Hugging Face credentials are intentionally excluded from this
repository.

## Installation

Create a separate Python environment for each backbone because the required
Diffusers and Transformers versions differ. Install a PyTorch and torchvision
pair compatible with the local CUDA driver first, then install the
model-specific requirements.

```bash
git clone https://github.com/Daydream0819/TFDG.git
cd TFDG

python -m venv .venv-sd35
source .venv-sd35/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r code/sd35/requirements.txt
```

Replace `cu124` with the CUDA wheel index matching the target machine. See
the README in the chosen model folder for its checkpoint download command and
its runner options. SD3.5-Large requires an accepted Hugging Face license and
an environment variable named `HF_TOKEN`.

## Quick start

After preparing the SD3.5-Large checkpoint, a single paired sample can be
generated as follows:

```bash
PYTHONPATH=code python code/sd35/run.py \
  --model /path/to/sd35-large \
  --prompt "A kingfisher perched on a branch" \
  --seeds 0 \
  --steps 20 \
  --scales 15 \
  --methods tfdg \
  --outdir runs/sd35_tfdg
```

Use the same prompt, seed, solver, number of steps, and terminal guidance
scale for all compared methods. For formal experiments, pass an immutable
JSON manifest through the `--manifest` argument.

## Evaluation and tests

Evaluation utilities for text-to-image and class-conditional experiments are
in `code/evaluation/`. They download their metric weights into a user-chosen
cache directory and never require those weights to be committed.

The shared guidance implementation can be checked with:

```bash
PYTHONPATH=code python -m unittest discover -s code/tests -v
```

## Repository structure

```text
code/
  common/       shared parameterization conversion and guidance rules
  sd15/         Stable Diffusion v1.5 adapter
  sd35/         Stable Diffusion 3.5 Large adapter and Figure 1 trajectory tool
  dit/          DiT-XL/2 adapter
  edm2/         EDM2-S adapter
  evaluation/   fidelity, preference, and class-conditional metrics
  tests/        unit tests for shared guidance behavior
```

## License

This code is released under the MIT License. The pretrained models and
external repositories downloaded by the scripts retain their own licenses and
terms of use.
