"""demo_app.py -- segment a particle with a click, and see it classified.

A deliberately small Gradio page showing the two halves of the paper's
pipeline joined end to end, on 60 easy held-out test images (20 Fiber,
20 Cluster, 20 Matrix; see demo_images.csv and select_demo_images.py):

    click on a particle  ->  SAM (ViT-B) proposes a mask from the point
                         ->  DINOv2 hypercolumn features are pooled INSIDE
                             that mask and the paper's MLP head predicts
                             Fiber / Cluster / Matrix / MatrixSurface

Left-click adds a positive point (this is the particle), right-click... is
not available in Gradio, so use the radio button to switch a click to
negative (this is background).  Every click re-runs SAM on all points so
far and re-classifies the mask.  The true class is revealed only after the
first prediction, and the image is never shown to the classifier as a whole:
it sees only the mask you drew.

Prerequisites (see demo/README.md):
    python fetch_demo_images.py [--download]
    python ../download_sam_weights.py --model vit_b       (358 MB, once)
    python demo_app.py
"""
from __future__ import annotations
import csv, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "classification"))
sys.path.insert(0, str(REPO / "sem_particle_analysis"))
sys.path.insert(0, str(HERE))

# the segmentation package prints a few unicode glyphs; a cp1252 console would choke
for _st in (sys.stdout, sys.stderr):
    try:
        _st.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import gradio as gr
from PIL import Image

from classify import CNTClassifier, CLASSES

COLOURS = {"Fiber": (200, 60, 60), "Cluster": (60, 160, 60),
           "Matrix": (60, 90, 200), "MatrixSurface": (170, 90, 200)}
DISPLAY = 1024                    # images are 2048 px; shown at half size


# ----------------------------------------------------------------- models
def sam_checkpoint():
    import os
    cands = [os.environ.get("SAM_CHECKPOINT"),
             REPO / "sam_weights" / "sam_vit_b_01ec64.pth"]
    for c in cands:
        if c and Path(c).is_file():
            return str(c)
    raise SystemExit("SAM ViT-B checkpoint not found.  Run\n"
                     "    python download_sam_weights.py --model vit_b\n"
                     "at the repository root (358 MB), or set SAM_CHECKPOINT.")


print("loading SAM ViT-B ...")
from sem_particle_analysis.model import SAMModel
SAM = SAMModel(sam_checkpoint(), model_type="vit_b")
print("loading DINOv2 + classifier head ...")
CLF = CNTClassifier()


# ----------------------------------------------------------------- data
rows = list(csv.DictReader(open(HERE / "demo_images.csv")))
IMAGES = {}
for r in rows:
    p = HERE / "images" / r["category"] / r["filename"]
    if p.exists():
        IMAGES[f"{r['category']}  |  {r['filename']}"] = (p, r["category"])
if not IMAGES:
    raise SystemExit("no demo images found under demo/images/.  Run\n"
                     "    python fetch_demo_images.py          (from a local dataset copy)\n"
                     "    python fetch_demo_images.py --download   (from Harvard Dataverse)")
missing = len(rows) - len(IMAGES)
if missing:
    print(f"note: {missing} of {len(rows)} demo images are missing; run fetch_demo_images.py")

# The list shown in the dropdown is shuffled so the class is not given away by
# position; the label is part of the key only so the user can find an image again.
rng = np.random.default_rng(0)
CHOICES = [k.split("|")[1].strip() for k in IMAGES]
KEY_OF = {k.split("|")[1].strip(): k for k in IMAGES}
rng.shuffle(CHOICES)


class S:                                   # one session state per browser tab
    def __init__(self):
        self.key = None; self.gray = None; self.rgb = None
        self.points = []; self.labels = []; self.mask = None; self.encoded = False


def load(name, s):
    if not name:
        return None, "", s
    s = S(); s.key = KEY_OF[name]
    p, _ = IMAGES[s.key]
    im = Image.open(p).convert("L")
    s.gray = np.asarray(im)
    s.rgb = np.stack([s.gray] * 3, -1)
    SAM.set_image(s.rgb); s.encoded = True
    return render(s), "Click on a particle.", s


def render(s):
    disp = Image.fromarray(s.rgb).resize((DISPLAY, DISPLAY), Image.BILINEAR)
    out = np.asarray(disp).astype(np.float32)
    if s.mask is not None and s.mask.any():
        m = np.asarray(Image.fromarray(s.mask.astype(np.uint8) * 255)
                       .resize((DISPLAY, DISPLAY), Image.NEAREST)) > 127
        col = np.array(COLOURS.get(s.pred, (255, 200, 0)), np.float32) if hasattr(s, "pred") \
              else np.array((255, 200, 0), np.float32)
        out[m] = 0.55 * out[m] + 0.45 * col
        # outline
        try:
            from skimage.segmentation import find_boundaries
            b = find_boundaries(m, mode="outer")
            out[b] = col
        except Exception:
            pass
    out = out.clip(0, 255).astype(np.uint8)
    im = Image.fromarray(out)
    from PIL import ImageDraw
    d = ImageDraw.Draw(im)
    sc = DISPLAY / s.gray.shape[1]
    for (x, y), l in zip(s.points, s.labels):
        cx, cy = x * sc, y * sc
        d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7],
                  fill=(60, 220, 60) if l == 1 else (230, 50, 50), outline="white", width=2)
    return im


def click(mode, s, evt: gr.SelectData):
    if s is None or s.gray is None:
        return None, "Choose an image first.", s
    x, y = evt.index
    sc = s.gray.shape[1] / DISPLAY
    s.points.append((x * sc, y * sc)); s.labels.append(1 if mode.startswith("Positive") else 0)
    t0 = time.time()
    masks, scores, _ = SAM.predict(point_coords=np.array(s.points, float),
                                   point_labels=np.array(s.labels, int),
                                   multimask_output=True)
    s.mask = masks[int(np.argmax(scores))].astype(bool)
    t_sam = time.time() - t0
    if not s.mask.any():
        s.pred = None
        return render(s), "SAM returned an empty mask; try another point.", s
    t0 = time.time()
    label, probs = CLF.predict(s.gray, s.mask)
    t_clf = time.time() - t0
    s.pred = label
    _, true = IMAGES[s.key]
    area = int(s.mask.sum())
    bars = "\n".join(f"  {c:14} {'#' * int(round(40 * p)):40} {100*p:5.1f}%"
                     for c, p in zip(CLASSES, probs))
    verdict = "correct" if label == true else "WRONG"
    msg = (f"prediction: **{label}**   (true class: {true} -> {verdict})\n\n"
           f"```\n{bars}\n```\n"
           f"mask {area:,} px ({100*area/s.mask.size:.1f}% of the frame) from "
           f"{len(s.points)} point(s); SAM {t_sam*1000:.0f} ms, DINOv2 + head {t_clf*1000:.0f} ms")
    return render(s), msg, s


def undo(s):
    if s is None or not s.points:
        return (render(s) if s and s.gray is not None else None), "Nothing to undo.", s
    s.points.pop(); s.labels.pop(); s.mask = None; s.pred = None
    if s.points:
        masks, scores, _ = SAM.predict(point_coords=np.array(s.points, float),
                                       point_labels=np.array(s.labels, int),
                                       multimask_output=True)
        s.mask = masks[int(np.argmax(scores))].astype(bool)
        if s.mask.any():
            s.pred, _ = CLF.predict(s.gray, s.mask)
    return render(s), f"{len(s.points)} point(s) left.", s


def reset(s):
    if s is None or s.gray is None:
        return None, "", s
    s.points, s.labels, s.mask, s.pred = [], [], None, None
    return render(s), "Click on a particle.", s


with gr.Blocks(title="CNT segment + classify demo") as app:
    gr.Markdown(
        "## Segment a carbon-nanotube particle, and see it classified\n"
        "Pick a micrograph, click on a particle.  SAM proposes a mask from your click; "
        "DINOv2 features pooled **inside that mask** go to the paper's MLP head.  "
        f"{len(IMAGES)} held-out test images (Fiber / Cluster / Matrix), none seen in training.  "
        "The head is a single refit of the pre-specified configuration "
        f"({CLF.meta['test_acc']:.1f}% on the full 179-image test set); "
        "the paper's headline number is the five-fold ensemble.")
    st = gr.State(S())
    with gr.Row():
        with gr.Column(scale=3):
            sel = gr.Dropdown(CHOICES, label="micrograph (held-out test set, order shuffled)")
            img = gr.Image(type="pil", label="click on a particle", interactive=False,
                           height=DISPLAY * 0.75)
        with gr.Column(scale=2):
            mode = gr.Radio(["Positive point (this is the particle)",
                             "Negative point (this is background)"],
                            value="Positive point (this is the particle)", label="next click")
            with gr.Row():
                b_undo = gr.Button("undo last point"); b_reset = gr.Button("reset")
            out = gr.Markdown("")
    sel.change(load, [sel, st], [img, out, st])
    img.select(click, [mode, st], [img, out, st])
    b_undo.click(undo, [st], [img, out, st])
    b_reset.click(reset, [st], [img, out, st])

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=7861)
    a = ap.parse_args()
    app.launch(server_name=a.host, server_port=a.port, show_error=True)
