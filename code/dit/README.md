# DiT-XL/2

This folder contains only the epsilon-prediction adapter for DiT-XL/2. Install
PyTorch with the CUDA version provided by the server, then install the
remaining dependencies with `pip install -r requirements.txt`. Prepare the
official repository, checkpoint, and VAE in one command:

```bash
python download_model.py
```

The command clones the official DiT repository into `vendor/DiT/`, downloads
`weights/DiT-XL-2-256x256.pt`, and downloads the Diffusers VAE into
`weights/vae/`. Use `--skip-repo`, `--skip-checkpoint`, or `--skip-vae` when an
asset is already provided on the data disk.
The runner maps

```text
epsilon -> clean latent anchor -> guidance -> epsilon -> DDPM update
```

Only the selected epsilon channels are guided; learned variance remains the
conditional prediction. The full, unsharded sample list is recorded in one
manifest, while each worker processes only its assigned shard.

The final guidance scale and TFDG parameters are not yet frozen; see
`docs/PROTOCOL_DECISIONS.md`.
