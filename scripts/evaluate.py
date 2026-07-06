"""
Car Damage Detection — Model Evaluation
Developer: Saksham Pathak (github.com/parthmax2)

Runs YOLO validation on a chosen split and reports per-class metrics,
overall mAP, and an optional inference speed benchmark.

Usage (run from project root):
  python scripts/evaluate.py                          # test split, best.pt
  python scripts/evaluate.py --model runs/phase2_full/weights/best.pt
  python scripts/evaluate.py --split val
  python scripts/evaluate.py --save-images            # save annotated predictions
  python scripts/evaluate.py --speed                  # add latency benchmark
  python scripts/evaluate.py --conf 0.25 --iou 0.5   # custom NMS thresholds
"""

import argparse
import time
from pathlib import Path

from ultralytics import YOLO

# ── project root ─────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
DATA_YAML     = str(ROOT / "configs" / "data.yaml")
DEFAULT_MODEL = str(ROOT / "best.pt")
RUNS_DIR      = str(ROOT / "runs")

CLASS_NAMES = [
    "no_damage", "lost_parts", "torn", "dent",
    "paint_scratch", "hole", "broken_glass", "broken_lamp",
]


# ─────────────────────────────────────────────
def run_validation(
    model_path: str,
    split: str,
    conf: float,
    iou: float,
    save_images: bool,
):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Weights not found: {model_path}")
    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(f"Dataset config not found: {DATA_YAML}")

    print(f"\n{'='*60}")
    print(f"  Model  : {model_path}")
    print(f"  Split  : {split}")
    print(f"  Conf   : {conf}   IoU: {iou}")
    print(f"{'='*60}")

    model   = YOLO(model_path)
    metrics = model.val(
        data        = DATA_YAML,
        split       = split,
        conf        = conf,
        iou         = iou,
        save        = save_images,
        save_txt    = False,
        save_json   = False,
        plots       = True,
        project     = RUNS_DIR,
        name        = f"eval_{split}",
        exist_ok    = True,
        verbose     = True,
    )

    _print_results(metrics)
    return metrics


# ─────────────────────────────────────────────
def _print_results(metrics) -> None:
    box = metrics.box

    print(f"\n{'='*60}")
    print("  OVERALL METRICS")
    print(f"{'='*60}")
    print(f"  mAP @ 0.5        : {box.map50:.4f}")
    print(f"  mAP @ 0.5:0.95   : {box.map:.4f}")
    print(f"  Precision (mean) : {box.mp:.4f}")
    print(f"  Recall    (mean) : {box.mr:.4f}")

    print(f"\n{'='*60}")
    print("  PER-CLASS METRICS")
    print(f"{'='*60}")
    header = f"  {'Class':<20} {'AP@0.5':>8} {'mAP':>8} {'P':>8} {'R':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for i, name in enumerate(CLASS_NAMES):
        ap50 = box.ap50[i] if i < len(box.ap50) else float("nan")
        mAP  = box.maps[i] if i < len(box.maps) else float("nan")
        p    = box.p[i]    if i < len(box.p)    else float("nan")
        r    = box.r[i]    if i < len(box.r)    else float("nan")
        print(f"  {name:<20} {ap50:>8.4f} {mAP:>8.4f} {p:>8.4f} {r:>8.4f}")

    print(f"{'='*60}")
    print(f"  Confusion matrix and plots saved to: {RUNS_DIR}/eval_*/\n")


# ─────────────────────────────────────────────
def speed_benchmark(model_path: str, n: int = 100) -> None:
    import torch
    from PIL import Image

    model = YOLO(model_path)
    dummy = Image.new("RGB", (640, 640), color=(128, 128, 128))

    for _ in range(10):              # warmup
        model(dummy, verbose=False)

    t0 = time.perf_counter()
    for _ in range(n):
        model(dummy, verbose=False)
    elapsed = time.perf_counter() - t0

    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"\n[speed] {n} × 640×640 inference on {device}")
    print(f"  Avg latency : {elapsed / n * 1000:.1f} ms / image")
    print(f"  Throughput  : {n / elapsed:.1f} FPS")


# ─────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a car-damage YOLO model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",       default=DEFAULT_MODEL,
                        help="Path to model weights (.pt)")
    parser.add_argument("--split",       default="test",
                        choices=["train", "val", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--conf",        type=float, default=0.001,
                        help="Confidence threshold (use 0.001 for full mAP curve)")
    parser.add_argument("--iou",         type=float, default=0.6,
                        help="IoU threshold for NMS")
    parser.add_argument("--save-images", action="store_true",
                        help="Save annotated prediction images")
    parser.add_argument("--speed",       action="store_true",
                        help="Run inference speed benchmark after evaluation")
    args = parser.parse_args()

    run_validation(
        model_path  = args.model,
        split       = args.split,
        conf        = args.conf,
        iou         = args.iou,
        save_images = args.save_images,
    )

    if args.speed:
        speed_benchmark(args.model)


if __name__ == "__main__":
    main()
