"""
Car Damage Detection — YOLO11 Fine-tuning Script
Developer: Saksham Pathak (github.com/parthmax2)

Strategy: two-phase transfer learning
  Phase 1 — freeze backbone, train detection head only  (fast convergence)
  Phase 2 — unfreeze all layers, fine-tune end-to-end   (accuracy gain)

Usage (run from project root):
  python scripts/train.py                         # full two-phase training
  python scripts/train.py --phase 1               # head only
  python scripts/train.py --phase 2               # fine-tune (needs phase-1 output)
  python scripts/train.py --resume runs/phase1_head/weights/last.pt
  python scripts/train.py --model yolo11m.pt      # larger backbone
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

# ── resolve paths relative to project root ───────────────────────
ROOT      = Path(__file__).resolve().parent.parent
DATA_YAML = str(ROOT / "configs" / "data.yaml")
RUNS_DIR  = str(ROOT / "runs")
FINAL_PT  = ROOT / "best.pt"

# ── base model choices (all pretrained on COCO) ──────────────────
#   yolo11n.pt  nano   ~2.6 M   fastest inference
#   yolo11s.pt  small  ~9.4 M   balanced  ← default
#   yolo11m.pt  medium ~20  M
#   yolo11l.pt  large  ~25  M
#   yolo11x.pt  xlarge ~56  M   highest accuracy
DEFAULT_BASE = "yolo11s.pt"

# ── Phase 1 — head training (backbone frozen) ────────────────────
PHASE1 = dict(
    epochs           = 30,
    imgsz            = 640,
    batch            = 16,
    lr0              = 1e-3,
    lrf              = 0.01,
    momentum         = 0.937,
    weight_decay     = 5e-4,
    warmup_epochs    = 3,
    warmup_momentum  = 0.8,
    warmup_bias_lr   = 0.1,
    freeze           = 10,        # freeze first 10 backbone layers
    # augmentation
    hsv_h            = 0.015,
    hsv_s            = 0.7,
    hsv_v            = 0.4,
    degrees          = 10.0,
    translate        = 0.1,
    scale            = 0.5,
    shear            = 2.0,
    flipud           = 0.0,
    fliplr           = 0.5,
    mosaic           = 1.0,
    mixup            = 0.1,
    copy_paste       = 0.1,
    # logging
    project          = RUNS_DIR,
    name             = "phase1_head",
    exist_ok         = True,
    save_period      = 5,
    patience         = 10,
    val              = True,
    plots            = True,
)

# ── Phase 2 — full fine-tune (all layers unfrozen) ───────────────
PHASE2 = dict(
    epochs           = 50,
    imgsz            = 640,
    batch            = 8,         # smaller batch: all layers need memory
    lr0              = 1e-4,      # much lower LR to avoid destroying phase-1 weights
    lrf              = 0.01,
    momentum         = 0.937,
    weight_decay     = 5e-4,
    warmup_epochs    = 2,
    warmup_momentum  = 0.8,
    warmup_bias_lr   = 0.01,
    freeze           = 0,
    # lighter augmentation than phase 1
    hsv_h            = 0.015,
    hsv_s            = 0.7,
    hsv_v            = 0.4,
    degrees          = 5.0,
    translate        = 0.1,
    scale            = 0.5,
    shear            = 0.0,
    flipud           = 0.0,
    fliplr           = 0.5,
    mosaic           = 0.9,
    mixup            = 0.05,
    copy_paste       = 0.05,
    # logging
    project          = RUNS_DIR,
    name             = "phase2_full",
    exist_ok         = True,
    save_period      = 5,
    patience         = 15,
    val              = True,
    plots            = True,
)


# ─────────────────────────────────────────────
def phase1(base_model: str) -> Path:
    print("\n" + "=" * 60)
    print("PHASE 1 — Training detection head  (backbone frozen)")
    print("=" * 60)
    model   = YOLO(base_model)
    result  = model.train(data=DATA_YAML, **PHASE1)
    best    = Path(result.save_dir) / "weights" / "best.pt"
    print(f"[phase1] Best checkpoint: {best}")
    return best


def phase2(checkpoint: Path) -> Path:
    print("\n" + "=" * 60)
    print("PHASE 2 — Full model fine-tune  (all layers unfrozen)")
    print("=" * 60)
    model   = YOLO(str(checkpoint))
    result  = model.train(data=DATA_YAML, **PHASE2)
    best    = Path(result.save_dir) / "weights" / "best.pt"
    print(f"[phase2] Best checkpoint: {best}")
    return best


def resume(run_path: str) -> None:
    model = YOLO(run_path)
    model.train(resume=True)


def export_final(checkpoint: Path) -> None:
    shutil.copy2(checkpoint, FINAL_PT)
    print(f"\n[export] Final weights  → {FINAL_PT.resolve()}")
    model = YOLO(str(checkpoint))
    model.export(format="onnx", imgsz=640, simplify=True)
    print("[export] ONNX model exported alongside checkpoint.")


# ─────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-phase YOLO11 fine-tuning for car damage detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",  default=DEFAULT_BASE,
                        help="Base YOLO11 pretrained weights")
    parser.add_argument("--phase",  type=int, choices=[1, 2],
                        help="Run only phase 1 or 2 (default: both)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to last.pt or run dir to resume an interrupted run")
    parser.add_argument("--no-export", action="store_true",
                        help="Skip copying best.pt to project root after training")
    args = parser.parse_args()

    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(
            f"{DATA_YAML} not found.\n"
            "Run scripts/prepare_data.py first."
        )

    if args.resume:
        resume(args.resume)
        return

    if args.phase == 1:
        best = phase1(args.model)
    elif args.phase == 2:
        p1_best = ROOT / "runs" / "phase1_head" / "weights" / "best.pt"
        if not p1_best.exists():
            raise FileNotFoundError(
                f"Phase-1 checkpoint not found at {p1_best}.\n"
                "Run --phase 1 first."
            )
        best = phase2(p1_best)
    else:
        p1_best = phase1(args.model)
        best    = phase2(p1_best)

    if not args.no_export:
        export_final(best)

    print("\n✓ Training complete.")
    print(f"  Logs / plots  →  {RUNS_DIR}/")
    print(f"  Final model   →  {FINAL_PT}")
    print("\nTo launch the Gradio app:  python app.py")


if __name__ == "__main__":
    main()
