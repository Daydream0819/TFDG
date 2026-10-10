# EDM2-S

This folder contains only the direct-clean-prediction adapter for EDM2-S. Install
PyTorch with the CUDA version provided by the server, then install the
remaining dependencies with `pip install -r requirements.txt`. Prepare the
official repository and the guided S/XS checkpoints with:

```bash
python download_model.py
export DNNLIB_CACHE_DIR="$PWD/weights/cache"
```

The command clones the official EDM2 repository into `vendor/edm2/` and stores
the conditional S checkpoint and unconditional XS guide checkpoint in
`weights/cache/`. The runner automatically checks this cache before attempting
another network download.

The paper uses the conditional EDM2-S checkpoint and its unconditional guide
network. A formal run must pass `--scales` explicitly: the official
`edm2-img512-s-guid-dino` preset defaults to 1.9, whereas the current manuscript
draft describes a strong-guidance experiment at 4.

EDM2 output filenames are seed-based, so manifest seeds must be globally
unique. The full sample list is stored in the shared run manifest before
worker sharding.
