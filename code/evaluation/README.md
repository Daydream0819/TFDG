# Frozen evaluators

## Evaluation environment and weights

Install PyTorch and torchvision with versions compatible with the server CUDA
driver first, then install the metric packages:

```bash
pip install -r requirements.txt
```

Warm all metric checkpoints into one explicit cache before evaluating:

```bash
python download_weights.py \
  --cache-dir /autodl-tmp/tfdg_eval_cache \
  --device cpu
```

This downloads OpenCLIP ViT-H/14 (`laion2b_s32b_b79k`), Clean-FID's Inception
network, torchvision Inception-v3 and ResNet-50, ImageReward-v1.0, the selected
HPSv2 checkpoint, and PickScore. Set `HF_TOKEN` when a Hugging Face model
requires authentication. Use `--skip-preference`, `--skip-clip`, or
`--skip-fid` to prepare only a subset.

`text_to_image.py` evaluates the nested SD3.5 output layout directly and
requires an exact one-to-one match with the frozen manifest. It reports
CleanFID, CLIPScore, and paired appearance statistics.

`preference.py` reports ImageReward, HPSv2, and PickScore over the same exact
sample set. `merge_text_metrics.py` joins the two tables.

`class_conditional.py` evaluates DiT or EDM2 outputs in one fixed torchvision
Inception feature space and computes the union-of-kNN-balls definition of
improved Precision/Recall. DiT uses `--filename-mode sample_id`; EDM2 uses
`--filename-mode seed`. Pilot class-conditional P/R values produced by the old
nearest-center implementation must not be reused.
