"""
Car Damage Detection — Dataset Preparation
Developer: Saksham Pathak (github.com/parthmax2)

Supports two modes:
  1. Roboflow download  →  python scripts/prepare_data.py --roboflow
  2. Organise local data →  python scripts/prepare_data.py --source path/to/raw

Raw folder expected layout (mode 2):
  <source>/
  ├── images/   (*.jpg / *.png / *.bmp / *.webp)
  └── labels/   (*.txt  YOLO format — class cx cy w h, normalised)

Output layout written to <dest> (default: data/ at project root):
  data/
  ├── images/train | val | test
  └── labels/train | val | test
"""

import os
import random
import shutil
import argparse
from collections import Counter
from pathlib import Path

# ── project root (one level above this script) ───────────────────
ROOT = Path(__file__).resolve().parent.parent

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
TEST_RATIO  = 0.10
SEED        = 42

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_NAMES = [
    "no_damage", "lost_parts", "torn", "dent",
    "paint_scratch", "hole", "broken_glass", "broken_lamp",
]


# ─────────────────────────────────────────────
# 1. ROBOFLOW DOWNLOAD
# ─────────────────────────────────────────────
def download_from_roboflow(dest: Path) -> None:
    """
    Downloads a car-damage dataset from Roboflow Universe.
    Requires:  pip install roboflow
    Free API key: https://roboflow.com

    Dataset used:
      https://universe.roboflow.com/car-damage-detection/car-damage-detection-dataset
    Swap workspace / project / version to match your own export.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        raise SystemExit("roboflow not installed. Run: pip install roboflow")

    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        raise SystemExit(
            "Set ROBOFLOW_API_KEY before running with --roboflow.\n"
            "  Windows PowerShell: $env:ROBOFLOW_API_KEY='YOUR_KEY'\n"
            "  Linux / macOS:      export ROBOFLOW_API_KEY=YOUR_KEY"
        )

    rf      = Roboflow(api_key=api_key)
    project = rf.workspace("car-damage-detection").project("car-damage-detection-dataset")
    dataset = project.version(1).download("yolov11", location=str(dest / "_rf_tmp"))

    _remap_roboflow_export(dest / "_rf_tmp", dest)
    shutil.rmtree(dest / "_rf_tmp", ignore_errors=True)
    print("[prepare] Roboflow download complete.")


def _remap_roboflow_export(src: Path, dst: Path) -> None:
    for rf_split, out_split in [("train", "train"), ("valid", "val"), ("test", "test")]:
        for kind in ("images", "labels"):
            src_dir = src / rf_split / kind
            dst_dir = dst / kind / out_split
            dst_dir.mkdir(parents=True, exist_ok=True)
            if src_dir.exists():
                for f in src_dir.iterdir():
                    shutil.copy2(f, dst_dir / f.name)


# ─────────────────────────────────────────────
# 2. LOCAL DATA ORGANISATION
# ─────────────────────────────────────────────
def organise_local(source: Path, dest: Path) -> None:
    img_dir = source / "images"
    lbl_dir = source / "labels"

    if not img_dir.exists():
        raise SystemExit(f"images/ directory not found inside {source}")
    if not lbl_dir.exists():
        raise SystemExit(f"labels/ directory not found inside {source}")

    all_images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    print(f"[prepare] Found {len(all_images)} images in {img_dir}")

    paired, no_label = [], []
    for img in all_images:
        lbl = lbl_dir / (img.stem + ".txt")
        (paired if lbl.exists() else no_label).append((img, lbl) if lbl.exists() else img)

    if no_label:
        print(f"[prepare] WARNING – {len(no_label)} images skipped (no matching label)")

    random.seed(SEED)
    random.shuffle(paired)

    n       = len(paired)
    n_val   = int(n * VAL_RATIO)
    n_test  = int(n * TEST_RATIO)
    n_train = n - n_val - n_test

    splits = {
        "train": paired[:n_train],
        "val":   paired[n_train : n_train + n_val],
        "test":  paired[n_train + n_val :],
    }

    for split, items in splits.items():
        img_out = dest / "images" / split
        lbl_out = dest / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img, lbl in items:
            shutil.copy2(img, img_out / img.name)
            shutil.copy2(lbl, lbl_out / lbl.name)
        print(f"[prepare] {split:5s} → {len(items):5d} samples")

    print(f"[prepare] Dataset written to: {dest.resolve()}")


# ─────────────────────────────────────────────
# 3. INTEGRITY VALIDATION
# ─────────────────────────────────────────────
def validate_dataset(dest: Path) -> None:
    print("\n[validate] Checking dataset integrity …")
    nc     = len(CLASS_NAMES)
    issues = 0

    for split in ("train", "val", "test"):
        img_dir = dest / "images" / split
        lbl_dir = dest / "labels" / split
        if not img_dir.exists():
            print(f"  MISSING: {img_dir}")
            continue

        img_stems = {p.stem for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
        lbl_stems = {p.stem for p in lbl_dir.glob("*.txt")}

        for stem in img_stems - lbl_stems:
            print(f"  [{split}] no label for: {stem}")
            issues += 1
        for stem in lbl_stems - img_stems:
            print(f"  [{split}] orphan label: {stem}")
            issues += 1

        for lbl_file in lbl_dir.glob("*.txt"):
            for line in lbl_file.read_text().splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                if not (0 <= cls_id < nc):
                    print(f"  [{split}] bad class {cls_id} in {lbl_file.name}")
                    issues += 1

        print(f"  [{split}] images={len(img_stems)}  labels={len(lbl_stems)}")

    if issues == 0:
        print("[validate] Dataset is clean.")
    else:
        print(f"[validate] {issues} issue(s) found — fix before training.")


# ─────────────────────────────────────────────
# 4. CLASS DISTRIBUTION REPORT
# ─────────────────────────────────────────────
def print_stats(dest: Path) -> None:
    print("\n[stats] Class distribution per split:")
    counters = {s: Counter() for s in ("train", "val", "test")}

    for split in ("train", "val", "test"):
        lbl_dir = dest / "labels" / split
        if not lbl_dir.exists():
            continue
        for f in lbl_dir.glob("*.txt"):
            for line in f.read_text().splitlines():
                parts = line.strip().split()
                if parts:
                    counters[split][int(parts[0])] += 1

    header = f"{'Class':<20}" + "".join(f"{s:>8}" for s in ("train", "val", "test"))
    print(header)
    print("-" * len(header))
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<20}" + "".join(f"{counters[s][i]:>8}" for s in ("train", "val", "test")))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare car-damage dataset for YOLO training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--roboflow", action="store_true",
                       help="Download from Roboflow (requires ROBOFLOW_API_KEY)")
    group.add_argument("--source", type=Path,
                       help="Path to raw dataset with images/ and labels/ subdirs")
    parser.add_argument("--dest", type=Path, default=ROOT / "data",
                        help="Output directory for the organised dataset")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip integrity check after organising")
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)

    if args.roboflow:
        download_from_roboflow(args.dest)
    else:
        organise_local(args.source, args.dest)

    if not args.no_validate:
        validate_dataset(args.dest)
        print_stats(args.dest)


if __name__ == "__main__":
    main()
