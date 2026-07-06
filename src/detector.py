"""
Car Damage Detection — inference engine
Developer: Saksham Pathak (github.com/parthmax2)

Exposes:
  CLASS_NAMES   dict[int, str]
  CLASS_COLORS  dict[int, tuple]
  load_model()  → YOLO
  detect()      → (annotated_img, DataFrame, csv_path, txt_path)
"""

import tempfile

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ── class registry ───────────────────────────────────────────────
CLASS_NAMES: dict[int, str] = {
    0: "no damage",
    1: "lost parts",
    2: "torn",
    3: "dent",
    4: "paint scratch",
    5: "hole",
    6: "broken glass",
    7: "broken lamp",
}

CLASS_COLORS: dict[int, tuple] = {
    0: (128, 128, 128),   # grey
    1: (0,   128, 255),   # orange
    2: (255,   0, 255),   # magenta
    3: (0,   255,   0),   # green
    4: (0,   140, 255),   # deep orange
    5: (0,   255, 255),   # yellow
    6: (0,     0, 255),   # red
    7: (255, 165,   0),   # blue-ish
}

_model: YOLO | None = None


# ── model loader (singleton) ─────────────────────────────────────
def load_model(weights: str = "best.pt") -> YOLO:
    global _model
    if _model is None:
        print(f"[detector] Loading model from {weights} …")
        _model = YOLO(weights)
    return _model


# ── helpers ──────────────────────────────────────────────────────
def _xyxy_to_yolo(boxes: list, img_w: int, img_h: int) -> np.ndarray:
    if not boxes:
        return np.empty((0, 4))
    out = []
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        w  = (x2 - x1) / img_w
        h  = (y2 - y1) / img_h
        out.append([cx, cy, w, h])
    return np.array(out)


def _draw_box(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
              label: str, color: tuple) -> None:
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(y1 - 10, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


# ── main inference function ───────────────────────────────────────
def detect(
    image_pil,
    resize: bool,
    thresholds: dict[int, float],
    weights: str = "best.pt",
) -> tuple:
    """
    Parameters
    ----------
    image_pil   : PIL.Image
    resize      : bool — cap image at 1024 px before inference
    thresholds  : per-class confidence floor {class_id: min_conf}
    weights     : path to model weights file

    Returns
    -------
    annotated_img : np.ndarray  (H, W, 3)
    df            : pd.DataFrame
    csv_path      : str  — temp CSV file path
    txt_path      : str  — temp YOLO label file path
    """
    img = np.array(image_pil)

    if resize and max(img.shape[:2]) > 1024:
        img = cv2.resize(img, (1024, 1024))

    model  = load_model(weights)
    preds  = model(img, augment=True)

    raw_boxes, confs, class_ids = [], [], []
    for r in preds:
        raw_boxes.extend(r.boxes.xyxy.cpu().numpy().tolist())
        confs.extend(r.boxes.conf.cpu().numpy().tolist())
        class_ids.extend(r.boxes.cls.cpu().numpy().astype(int).tolist())

    img_h, img_w = img.shape[:2]
    yolo_boxes   = _xyxy_to_yolo(raw_boxes, img_w, img_h)

    annotated    = img.copy()
    summary_rows = []
    yolo_lines   = []

    for i, (box, conf, cls) in enumerate(zip(raw_boxes, confs, class_ids)):
        if conf < thresholds.get(cls, 0.25):
            continue

        x1, y1, x2, y2 = map(int, box)
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        label = f"{CLASS_NAMES[cls]} {conf:.2f}"

        _draw_box(annotated, x1, y1, x2, y2, label, color)

        yc = yolo_boxes[i]
        summary_rows.append({
            "Class":       CLASS_NAMES[cls],
            "Confidence":  round(conf, 3),
            "YOLO [cx, cy, w, h]": (
                f"[{yc[0]:.6f}, {yc[1]:.6f}, {yc[2]:.6f}, {yc[3]:.6f}]"
            ),
        })
        yolo_lines.append(f"{cls} {' '.join(f'{v:.6f}' for v in yc)}")

    df = pd.DataFrame(summary_rows)

    csv_path = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
    df.to_csv(csv_path, index=False)

    txt_path = tempfile.NamedTemporaryFile(delete=False, suffix=".txt").name
    with open(txt_path, "w") as fh:
        fh.write("\n".join(yolo_lines))

    return annotated, df, csv_path, txt_path
