# Developer: Saksham Pathak (github.com/parthmax2)
import gradio as gr

from src.detector import CLASS_NAMES, CLASS_COLORS, detect


# ── confidence defaults per class ────────────────────────────────
_DEFAULTS = {0: 0.90, 1: 0.26, 2: 0.05, 3: 0.05,
             4: 0.05, 5: 0.16, 6: 0.33, 7: 0.05}


def run(img, resize, t0, t1, t2, t3, t4, t5, t6, t7):
    thresholds = {0: t0, 1: t1, 2: t2, 3: t3,
                  4: t4, 5: t5, 6: t6, 7: t7}
    return detect(img, resize, thresholds)


# ── colour legend HTML ────────────────────────────────────────────
_legend = (
    "<h4>Class Colour Legend</h4>"
    "<div style='display:flex;flex-wrap:wrap;gap:1em;'>"
    + "".join(
        f"<div style='display:flex;align-items:center;gap:0.5em;'>"
        f"<div style='width:18px;height:18px;background:rgb{CLASS_COLORS[i]};"
        f"border:1px solid #555;border-radius:3px'></div>"
        f"<span>{CLASS_NAMES[i]}</span></div>"
        for i in CLASS_NAMES
    )
    + "</div>"
)


# ── UI ────────────────────────────────────────────────────────────
with gr.Blocks(title="Car Damage Detection") as demo:

    gr.Markdown(
        "# YOLOv11 — Vehicle Damage Detection\n"
        "Upload a vehicle image to detect and classify damage. "
        "Adjust per-class confidence sliders to control sensitivity."
    )

    with gr.Row():
        img_input = gr.Image(type="pil", label="Input Image")
        resize_cb = gr.Checkbox(
            label="Resize to 1024 px max before inference (faster on large images)",
            value=False,
        )

    gr.Markdown("### Confidence Thresholds by Class")
    with gr.Row():
        t0 = gr.Slider(0.0, 1.0, value=_DEFAULTS[0], label="No Damage")
        t1 = gr.Slider(0.0, 1.0, value=_DEFAULTS[1], label="Lost Parts")
        t2 = gr.Slider(0.0, 1.0, value=_DEFAULTS[2], label="Torn")
        t3 = gr.Slider(0.0, 1.0, value=_DEFAULTS[3], label="Dent")
    with gr.Row():
        t4 = gr.Slider(0.0, 1.0, value=_DEFAULTS[4], label="Paint Scratch")
        t5 = gr.Slider(0.0, 1.0, value=_DEFAULTS[5], label="Hole")
        t6 = gr.Slider(0.0, 1.0, value=_DEFAULTS[6], label="Broken Glass")
        t7 = gr.Slider(0.0, 1.0, value=_DEFAULTS[7], label="Broken Lamp")

    gr.HTML(_legend)
    run_btn = gr.Button("Run Detection", variant="primary")

    with gr.Tabs():
        with gr.Tab("Annotated Image"):
            img_out = gr.Image(type="numpy", label="Detections")
        with gr.Tab("Results"):
            table_out    = gr.Dataframe(label="Detection Summary")
            csv_out      = gr.File(label="Download CSV")
            yolo_txt_out = gr.File(label="Download YOLO Labels (.txt)")

    run_btn.click(
        fn=run,
        inputs=[img_input, resize_cb, t0, t1, t2, t3, t4, t5, t6, t7],
        outputs=[img_out, table_out, csv_out, yolo_txt_out],
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
