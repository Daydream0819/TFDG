#!/usr/bin/env python3
"""Download and validate the learned models used by the evaluation scripts.

This script does not generate images and does not evaluate a method.  It only
warms the local caches so that the seven-method evaluation can be run later
without downloading one model at a time.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


# Configure Hugging Face's current Xet transfer path before importing any
# model-loading library.  Set HF_XET_HIGH_PERFORMANCE=0 externally when the
# machine has less than 64 GB of system RAM.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--clip-model", default="ViT-H-14")
    parser.add_argument("--clip-pretrained", default="laion2b_s32b_b79k")
    parser.add_argument("--hps-version", default="v2.1")
    parser.add_argument("--skip-clip", action="store_true")
    parser.add_argument("--skip-fid", action="store_true")
    parser.add_argument("--skip-preference", action="store_true")
    parser.add_argument(
        "--imagereward-only",
        action="store_true",
        help="Preload ImageReward but skip the optional HPSv2 and PickScore models.",
    )
    parser.add_argument(
        "--skip-classifiers",
        action="store_true",
        help="Skip torchvision classifiers used only by class-conditional evaluation.",
    )
    return parser.parse_args()


def configure_cache(cache_dir: str | None) -> None:
    if not cache_dir:
        return
    root = os.path.abspath(os.path.expanduser(cache_dir))
    os.makedirs(root, exist_ok=True)
    os.environ.setdefault("HF_HOME", os.path.join(root, "huggingface"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(root, "huggingface/hub"))
    os.environ.setdefault("TORCH_HOME", os.path.join(root, "torch"))
    os.environ.setdefault("CLIP_HOME", os.path.join(root, "clip"))
    print(f"Using evaluation cache root: {root}")


def preload_clip(model_name: str, pretrained: str) -> None:
    # The text evaluator defaults to OpenCLIP.  ImageReward also imports the
    # separate OpenAI `clip` Python package, which is installed as a runtime
    # dependency but does not need to preload a second CLIPScore checkpoint.
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device="cpu",
    )
    del model
    print(f"[OK] OpenCLIP {model_name}/{pretrained}")


def preload_fid(device: str, include_classifiers: bool = True) -> None:
    from cleanfid.features import build_feature_extractor

    model = build_feature_extractor(
        mode="clean", device=device, use_dataparallel=False
    )
    del model
    print("[OK] Clean-FID Inception checkpoint")

    if not include_classifiers:
        return

    # The class-conditional evaluator uses torchvision's frozen Inception and
    # ResNet checkpoints. Download them here rather than during evaluation.
    from torchvision import models

    inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    classifier = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    del inception, classifier
    print("[OK] torchvision Inception-v3 and ResNet-50 checkpoints")


def preload_preference(hps_version: str, imagereward_only: bool = False) -> None:
    # ImageReward applies a small compatibility patch for current
    # transformers versions; reuse the patch used by the evaluator.
    from evaluation.preference import (
        patch_transformers_for_imagereward,
    )

    patch_transformers_for_imagereward()
    import ImageReward as image_reward

    image_model = image_reward.load("ImageReward-v1.0")
    del image_model
    print("[OK] ImageReward-v1.0")

    if imagereward_only:
        return

    from huggingface_hub import hf_hub_download
    from hpsv2.utils import hps_version_map

    def download_hf_file(repo_id: str, filename: str, attempts: int = 5) -> str:
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                return hf_hub_download(repo_id, filename)
            except Exception as error:
                last_error = error
                if attempt == attempts:
                    raise
                print(
                    f"Hugging Face download failed ({attempt}/{attempts}); "
                    "resuming after a short delay..."
                )
                time.sleep(10)
        raise last_error  # pragma: no cover

    checkpoint = download_hf_file(
        "xswu/HPSv2",
        hps_version_map[hps_version],
    )
    print(f"[OK] HPSv2 {hps_version}: {checkpoint}")

    # HPS_v2.1_compressed.pt is a complete half-precision ViT-H/14 state
    # dictionary (about 1.97 GB), not a residual adapter.  Constructing the
    # architecture without OpenCLIP's LAION initialization avoids an otherwise
    # redundant 3.94 GB download.  The strict load in the evaluator verifies
    # that this checkpoint provides every required model parameter.
    # hpsv2==1.2.0 omits the standard OpenCLIP BPE archive from some wheels.
    # Reuse the byte-identical vocabulary shipped by open-clip-torch before
    # importing the bundled HPS OpenCLIP fork.
    import hpsv2
    import open_clip

    bpe_source = Path(open_clip.__file__).resolve().parent / "bpe_simple_vocab_16e6.txt.gz"
    bpe_target = Path(hpsv2.__file__).resolve().parent / "src/open_clip/bpe_simple_vocab_16e6.txt.gz"
    if not bpe_target.is_file():
        if not bpe_source.is_file():
            raise FileNotFoundError(f"OpenCLIP BPE vocabulary is missing: {bpe_source}")
        bpe_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bpe_source, bpe_target)
        print(f"[OK] Restored HPSv2 BPE vocabulary: {bpe_target}")

    from hpsv2.src.open_clip import create_model_and_transforms

    hps_model, _, _ = create_model_and_transforms(
        "ViT-H-14",
        pretrained=None,
        precision="amp",
        device="cpu",
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False,
    )
    del hps_model
    print("[OK] HPSv2 ViT-H-14 architecture")

    from transformers import AutoModel, AutoProcessor

    pickscore_name = "yuvalkirstain/PickScore_v1"
    processor = AutoProcessor.from_pretrained(pickscore_name)
    pick_model = AutoModel.from_pretrained(pickscore_name)
    del processor, pick_model
    print(f"[OK] PickScore: {pickscore_name}")


def main() -> None:
    args = parse_args()
    configure_cache(args.cache_dir)
    if not args.skip_clip:
        preload_clip(args.clip_model, args.clip_pretrained)
    if not args.skip_fid:
        preload_fid(args.device, include_classifiers=not args.skip_classifiers)
    if not args.skip_preference:
        preload_preference(args.hps_version, imagereward_only=args.imagereward_only)
    print("All requested evaluation weights are ready in the local caches.")


if __name__ == "__main__":
    main()
