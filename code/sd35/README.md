# SD3.5-Large

This folder contains the velocity-prediction adapter for SD3.5-Large. Install
a PyTorch and torchvision pair compatible with the local CUDA driver, then
install the remaining dependencies from the repository root:

```bash
pip install -r code/sd35/requirements.txt
```

## Download checkpoint

SD3.5-Large is gated. Sign in to
[the official model page](https://huggingface.co/stabilityai/stable-diffusion-3.5-large),
accept its license, and create a Hugging Face access token with read permission.
From the repository root, run:

```bash
export HF_TOKEN='your_huggingface_token'

python code/sd35/download_model.py \
  --output weights/sd35-large
```

The command stores the Diffusers checkpoint at `weights/sd35-large/`. You may
place it on a data disk by changing `--output` and pass that directory to
`run.py` with `--model`. The runner maps

```text
velocity -> clean latent anchor -> guidance -> velocity -> Euler update
```

It writes `runs/sd35/<method>/w<scale>/<sample_id>.png` under the release root.
Use a fixed JSON manifest for formal paired evaluation.

The manuscript protocol is 20 Euler steps at terminal CFG scale 15. The final
TFDG radial parameters must be frozen in `docs/PROTOCOL_DECISIONS.md` before a
large run.

## Figure 1 visualization trajectory

`fig1_exp.py` runs one fixed prompt--seed condition through the complete TFDG
trajectory and saves all 21 latent states (the initial state plus 20 Euler
updates). It also dumps clean anchors, residual spectra, the three TFDG
components, their actual scheduled weights, the reconstructed guided anchor,
and the next sampler state at candidate steps 5, 7, 9, and 11. The candidate
step is selected only after inspecting the saved trajectory and components.

```bash
PYTHONPATH=code python code/sd35/fig1_exp.py \
  --model /root/autodl-tmp/hf_cache \
  --manifest code/sd35/fig1_visualization_manifest.json \
  --outdir /root/autodl-tmp/results/sd35-fig1-trajectory-macaw-v1 \
  --steps 20 --shift 3.0 --scale 15 --cutoff 0.30 \
  --low-min-cfg 1 --low-eta 0.5 \
  --high-min-cfg 3 --high-eta 1 \
  --candidate-steps 5 7 9 11 \
  --height 1024 --width 1024 --dtype bfloat16 --device cuda
```

The output directory is immutable: choose a new directory name for every
changed prompt, seed, or scientific setting.
