# SD3.5-Large

This folder contains only the velocity-prediction adapter for SD3.5-Large.
Install PyTorch with the CUDA version provided by the server, then install the
remaining dependencies with `pip install -r requirements.txt`. The gated
checkpoint is downloaded with:

```bash
export HF_TOKEN='your_huggingface_token'
python download_model.py
```

For a fresh AutoDL server, use the independent SD3.5 upload package under
`final_release/server_code/sd35/`. Its single `setup_and_download.sh` entry
point creates isolated generation/evaluation environments, accelerates package
installation, downloads the gated model, and warms the text-metric weights.

The command stores the Diffusers checkpoint at `weights/sd35-large/`. You may
use `--output` to place it on a data disk and pass that path to `run.py` with
`--model`. The runner maps

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
