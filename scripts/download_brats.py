#!/usr/bin/env python3
# scripts/download_brats.py
"""
BraTS Dataset Downloader — Kaggle API
ML Project by Rahul & Krishnaa for Dr. Valarmathi

Handles every BraTS version on Kaggle (2020-2024) with auto structure detection.

Usage:
  # First time: install kaggle credentials
  python scripts/download_brats.py --kaggle-json /path/to/kaggle.json --output ./data

  # Subsequent runs (credentials cached)
  python scripts/download_brats.py --output ./data

  # Pick specific dataset
  python scripts/download_brats.py --output ./data --dataset brats2020
"""

import os
import sys
import json
import shutil
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  KAGGLE DATASET REGISTRY
#  Ordered by reliability / community usage on Kaggle.
#  Each entry: (slug, display_name, version_tag)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASETS = [
    # ── BraTS 2020 — MOST RELIABLE, widely used, guaranteed structure ──
    # 369 training + 125 validation, 240x240x155, ~7.5 GB
    # Naming: BraTS20_Training_001/BraTS20_Training_001_{t1,t1ce,t2,flair,seg}.nii.gz
    ("awsaf49/brats20-dataset-training-validation", "BraTS 2020 Train+Val", "brats2020"),

    # ── BraTS 2021 — Largest single-file dataset on Kaggle ──
    # 1251 cases, 240x240x155, ~15 GB
    # Naming: BraTS2021_00000/BraTS2021_00000_{t1,t1ce,t2,flair,seg}.nii.gz
    ("dschettler8845/brats-2021-task1", "BraTS 2021 Task 1", "brats2021"),

    # ── BraTS 2023 — New naming convention ──
    # ~1470 GLI cases, 240x240x155
    # Naming: BraTS-GLI-XXXXX-YYY/BraTS-GLI-XXXXX-YYY-{t1n,t1c,t2w,t2f,seg}.nii.gz
    ("shakilrana/brats-2023-adult-glioma", "BraTS 2023 Adult Glioma", "brats2023"),

    # ── BraTS 2024 — Full GLI set ──
    # Naming same as 2023
    ("i212385nomanarif/2024-brats-glioma", "BraTS 2024 GLI (full)", "brats2024"),

    # ── BraTS 2024 — Small subset (fast download for testing) ──
    ("nguyenthanhkhanh/brats2024-small-dataset", "BraTS 2024 (small)", "brats2024"),

    # ── BraTS 2020 — Training only (smaller download) ──
    # 369 cases, ~5.5 GB
    ("awsaf49/brats2020-training-data", "BraTS 2020 Training Only", "brats2020"),

    # ── BraTS 2023 Pediatric ──
    ("mahamostafa/brats-2023-ped-dataset", "BraTS 2023 Pediatric", "brats2023"),
]


def install_kaggle_credentials(kaggle_json_path: str = None) -> bool:
    """
    Set up Kaggle API authentication. Priority:
      1. Explicit path argument
      2. KAGGLE_USERNAME + KAGGLE_KEY env vars
      3. Existing ~/.kaggle/kaggle.json
    """
    dest = Path.home() / ".kaggle" / "kaggle.json"

    if kaggle_json_path:
        src = Path(kaggle_json_path)
        if not src.exists():
            log.error(f"kaggle.json not found at {src}")
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dest))
        dest.chmod(0o600)
        log.info(f"Installed kaggle.json from {src}")
        return True

    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({
            "username": os.environ["KAGGLE_USERNAME"],
            "key": os.environ["KAGGLE_KEY"],
        }))
        dest.chmod(0o600)
        log.info("Created kaggle.json from environment variables")
        return True

    if dest.exists():
        dest.chmod(0o600)
        log.info(f"Using existing {dest}")
        return True

    log.error(
        "No Kaggle credentials found.\n"
        "  Option A: python download_brats.py --kaggle-json /path/to/kaggle.json\n"
        "  Option B: export KAGGLE_USERNAME=xxx KAGGLE_KEY=yyy\n"
        "  Option C: place kaggle.json at ~/.kaggle/kaggle.json\n"
        "  Get it from https://www.kaggle.com/settings → API → Create New Token"
    )
    return False


def kaggle_download(slug: str, dest_dir: str) -> bool:
    """Download + extract a Kaggle dataset. Returns True on success."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        log.info(f"  Downloading {slug} ...")
        api.dataset_download_files(slug, path=dest_dir, unzip=True)
        return True
    except Exception as e:
        log.warning(f"  Failed: {e}")
        return False


def discover_cases(root: str) -> list:
    """
    Recursively find all valid BraTS case dirs under `root`.
    A valid dir has at least one *seg* NIfTI and one modality NIfTI.
    """
    root_path = Path(root)
    seen = set()
    cases = []

    for nii in sorted(root_path.rglob("*.nii*")):
        if "seg" in nii.name.lower():
            d = nii.parent
            if d not in seen:
                seen.add(d)
                # Verify at least one non-seg NIfTI exists
                siblings = [f for f in d.glob("*.nii*") if "seg" not in f.name.lower()]
                if siblings:
                    cases.append(d)

    return sorted(cases, key=lambda p: p.name)


def detect_modalities(case_dir) -> dict:
    """
    Auto-detect modality files regardless of BraTS version.
    Returns: {'t1': path, 't1ce': path, 't2': path, 'flair': path, 'seg': path}

    Handles:
      BraTS2020: BraTS20_Training_001_{t1,t1ce,t2,flair,seg}.nii.gz
      BraTS2021: BraTS2021_00000_{t1,t1ce,t2,flair,seg}.nii.gz
      BraTS2023/24: BraTS-GLI-XXXXX-YYY-{t1n,t1c,t2w,t2f,seg}.nii.gz
    """
    case_dir = Path(case_dir)
    niftis = sorted(case_dir.glob("*.nii*"))

    result = {"t1": None, "t1ce": None, "t2": None, "flair": None, "seg": None}

    for f in niftis:
        n = f.name.lower()
        stem = f.stem.replace(".nii", "")  # handle .nii.gz double ext

        if "seg" in n:
            result["seg"] = str(f)

        # ── T1CE / T1C (must check before T1) ──
        elif any(n.endswith(x) for x in [
            "_t1ce.nii.gz", "_t1ce.nii", "-t1c.nii.gz", "-t1c.nii",
            "_t1gd.nii.gz", "-t1gd.nii.gz",
        ]):
            result["t1ce"] = str(f)

        # ── FLAIR / T2F (must check before T2) ──
        elif any(n.endswith(x) for x in [
            "_flair.nii.gz", "_flair.nii", "-t2f.nii.gz", "-t2f.nii",
        ]):
            result["flair"] = str(f)

        # ── T1 / T1N ──
        elif any(n.endswith(x) for x in [
            "_t1.nii.gz", "_t1.nii", "-t1n.nii.gz", "-t1n.nii",
        ]):
            result["t1"] = str(f)

        # ── T2 / T2W ──
        elif any(n.endswith(x) for x in [
            "_t2.nii.gz", "_t2.nii", "-t2w.nii.gz", "-t2w.nii",
        ]):
            result["t2"] = str(f)

    return result


def split_cases(cases, val_ratio=0.15, seed=42):
    """Deterministic train/val split."""
    import random
    rng = random.Random(seed)
    shuffled = list(cases)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def download_brats(output_dir="./data", dataset=None, kaggle_json=None, val_ratio=0.15):
    """
    Full pipeline: auth → download → discover → validate → split.
    Returns (train_dirs: list[str], val_dirs: list[str], version: str)
    """
    if not install_kaggle_credentials(kaggle_json):
        raise RuntimeError("Kaggle auth failed")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Filter dataset list if specific version requested
    if dataset:
        targets = [(s, n, v) for s, n, v in DATASETS if v == dataset]
        if not targets:
            targets = DATASETS  # fallback to all
    else:
        targets = DATASETS

    # Try each dataset
    version = None
    for slug, name, ver in targets:
        log.info(f"Trying: {name} ({slug})")
        if kaggle_download(slug, str(out)):
            version = ver
            log.info(f"✅ Downloaded: {name}")
            break

    if not version:
        raise RuntimeError("All downloads failed — check kaggle.json and internet")

    # Discover
    cases = discover_cases(str(out))
    log.info(f"Discovered {len(cases)} total case directories")

    # Validate
    valid = []
    for c in cases:
        mods = detect_modalities(c)
        n_mods = sum(1 for k in ["t1", "t1ce", "t2", "flair"] if mods[k])
        if mods["seg"] and n_mods >= 1:
            valid.append(str(c))
        else:
            log.debug(f"  Skipped {c.name}: seg={bool(mods['seg'])}, mods={n_mods}")

    log.info(f"Valid cases: {len(valid)}/{len(cases)}")
    if not valid:
        raise RuntimeError(f"No valid cases found in {out}")

    # Split
    train, val = split_cases(valid, val_ratio)
    log.info(f"Split: {len(train)} train / {len(val)} val")

    # Save manifest
    manifest = {
        "version": version,
        "total": len(valid),
        "train": len(train),
        "val": len(val),
        "train_dirs": train,
        "val_dirs": val,
    }
    manifest_path = out / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info(f"Manifest saved: {manifest_path}")

    return train, val, version


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download BraTS from Kaggle")
    p.add_argument("-o", "--output", default="./data")
    p.add_argument("-d", "--dataset", default=None,
                   choices=["brats2020", "brats2021", "brats2023", "brats2024"])
    p.add_argument("-k", "--kaggle-json", default=None)
    p.add_argument("--val-ratio", type=float, default=0.15)
    args = p.parse_args()

    train, val, ver = download_brats(args.output, args.dataset, args.kaggle_json, args.val_ratio)
    print(f"\n{'='*50}")
    print(f"  Dataset:    {ver}")
    print(f"  Train:      {len(train)} cases")
    print(f"  Validation: {len(val)} cases")
    print(f"  Location:   {args.output}")
    print(f"{'='*50}")
