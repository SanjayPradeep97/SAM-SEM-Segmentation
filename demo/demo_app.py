"""demo_app.py -- segment a particle with a click, and see it classified.

A small Gradio page showing the two halves of the paper's pipeline joined end
to end, on 60 held-out test images that ship with the repository (20 Fiber,
20 Cluster, 20 Matrix; see demo_images.csv, select_demo_images.py and
ATTRIBUTION.md):

    click on a particle  ->  SAM (ViT-B) proposes masks from the point(s)
                         ->  DINOv2 hypercolumn features are pooled INSIDE
                             the chosen mask and the paper's five-fold MLP
                             ensemble predicts Fiber / Cluster / Matrix /
                             MatrixSurface

Every click re-runs SAM on all points so far and re-classifies the mask.
"Show the expert mask" loads the mask from the paper's dataset for the same
image and classifies that instead, so the answer for a reference mask is
always one click away.  The classifier never sees the image as a whole: it
sees only the pixels inside the mask.

From click to mask (prompting.py).  SAM returns three candidate masks per
prompt; in TEM the particle is darker than the support film, so the demo
keeps the LARGEST candidate that (a) covers less than 60 % of the frame and
(b) is at least 30 % as dark, relative to its surroundings, as the darkest
candidate.  The "SAM's highest score" option keeps SAM's own choice for
comparison; demo/README.md has the measured numbers for both.

Prerequisites (see README.md):  run_demo.bat / run_demo.sh, or
    python ../download_sam_weights.py --model vit_b       (358 MB, once)
    python demo_app.py
"""
from __future__ import annotations
import csv, os, sys, time
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
from PIL import Image, ImageDraw

from classify import CNTClassifier, CLASSES
from prompting import choose_mask, MODE_RULE, MODE_SCORE

COLOURS = {"Fiber": (200, 60, 60), "Cluster": (60, 160, 60),
           "Matrix": (60, 90, 200), "MatrixSurface": (170, 90, 200)}
DISPLAY_W = 1024                  # images are 1350 x 1040; shown at this width


# ----------------------------------------------------------------- models
def sam_checkpoint():
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
print("loading DINOv2 + the classifier heads ...")
CLF = CNTClassifier()


# ----------------------------------------------------------------- data
ROWS = list(csv.DictReader(open(HERE / "demo_images.csv")))
IMAGES = {}                                       # stem -> (png path, mask path, class)
for r in ROWS:
    stem = Path(r["filename"]).stem
    p = HERE / "images" / r["category"] / f"{stem}.png"
    m = HERE / "masks" / f"{stem}_mask.png"
    if p.exists():
        IMAGES[stem] = (p, m if m.exists() else None, r["category"])
if not IMAGES:
    raise SystemExit(f"no demo images found under {HERE / 'images'} -- is the checkout complete?")
if len(IMAGES) < len(ROWS):
    print(f"note: {len(ROWS) - len(IMAGES)} of {len(ROWS)} demo images are missing")

# shown in a fixed shuffled order so the class is not given away by position
CHOICES = sorted(IMAGES)
np.random.default_rng(0).shuffle(CHOICES)


class S:                                   # one session state per browser tab
    def __init__(self):
        self.stem = None; self.gray = None; self.rgb = None
        self.points = []; self.labels = []; self.mask = None; self.pred = None
        self.source = ""                   # "clicks" or "expert"


def load(stem, s):
    s = S()
    if not stem:
        return None, "", s
    s.stem = stem
    p, _, _ = IMAGES[stem]
    s.gray = np.asarray(Image.open(p).convert("L"))
    s.rgb = np.stack([s.gray] * 3, -1)
    SAM.set_image(s.rgb)
    return render(s), "Click on a particle.", s


def render(s):
    h, w = s.gray.shape
    dh = int(round(h * DISPLAY_W / w))
    disp = Image.fromarray(s.rgb).resize((DISPLAY_W, dh), Image.BILINEAR)
    out = np.asarray(disp).astype(np.float32)
    if s.mask is not None and s.mask.any():
        m = np.asarray(Image.fromarray(s.mask.astype(np.uint8) * 255)
                       .resize((DISPLAY_W, dh), Image.NEAREST)) > 127
        col = np.array(COLOURS.get(s.pred, (255, 200, 0)), np.float32)
        out[m] = 0.55 * out[m] + 0.45 * col
        try:
            from skimage.segmentation import find_boundaries
            out[find_boundaries(m, mode="outer")] = col
        except Exception:
            pass
    im = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    sc = DISPLAY_W / w
    for (x, y), l in zip(s.points, s.labels):
        cx, cy = x * sc, y * sc
        d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7],
                  fill=(60, 220, 60) if l == 1 else (230, 50, 50), outline="white", width=2)
    return im


def classify_current(s, t_seg=None):
    """Classify s.mask, store the label, and return the report text."""
    t0 = time.time()
    label, probs = CLF.predict(s.gray, s.mask)
    t_clf = time.time() - t0
    s.pred = label
    _, _, true = IMAGES[s.stem]
    area = int(s.mask.sum())
    bars = "\n".join(f"  {c:14} {'#' * int(round(40 * p)):40} {100*p:5.1f}%"
                     for c, p in zip(CLASSES, probs))
    verdict = "correct" if label == true else "different from the expert label"
    how = (f"expert mask from the paper's dataset" if s.source == "expert"
           else f"SAM mask from {len(s.points)} point(s)")
    timing = (f"SAM {t_seg*1000:.0f} ms, " if t_seg is not None else "") + f"DINOv2 + heads {t_clf*1000:.0f} ms"
    warn = ""
    if s.source == "clicks":
        inside, outside = float(s.gray[s.mask].mean()), float(s.gray[~s.mask].mean())
        if inside >= outside:
            warn = ("\n\n**This mask is brighter than its surroundings, so it is probably support "
                    "film rather than a particle.** Reset and click on the dark material.")
        elif area > 0.5 * s.mask.size:
            warn = "\n\n**The mask covers most of the frame.** Add a negative point on the film to trim it."
    return (f"### {label}\n"
            f"expert label: **{true}** ({verdict})\n\n"
            f"```\n{bars}\n```\n"
            f"{how}: {area:,} px ({100*area/s.mask.size:.1f} % of the frame). {timing}.{warn}")


def run_sam(s, mode):
    t0 = time.time()
    masks, scores, _ = SAM.predict(point_coords=np.array(s.points, float),
                                   point_labels=np.array(s.labels, int),
                                   multimask_output=True)
    s.mask = choose_mask(masks, scores, s.gray, mode)
    return time.time() - t0


def click(mode, s, evt: gr.SelectData):
    if s is None or s.gray is None:
        return None, "Choose an image first.", s
    x, y = evt.index
    sc = s.gray.shape[1] / DISPLAY_W
    if s.source == "expert":                       # a click starts over from the expert mask
        s.points, s.labels = [], []
    s.source = "clicks"
    s.points.append((x * sc, y * sc)); s.labels.append(1 if mode_is_positive(mode) else 0)
    t_seg = run_sam(s, s.mask_mode)
    if not s.mask.any():
        s.pred = None
        return render(s), "SAM returned an empty mask; try another point.", s
    report = classify_current(s, t_seg)
    return render(s), report, s


def mode_is_positive(m):
    return str(m).startswith("Positive")


def set_mask_mode(mode, s):
    if s is None:
        s = S()
    s.mask_mode = mode
    if s.gray is not None and s.points and s.source == "clicks":
        t_seg = run_sam(s, mode)
        if s.mask.any():
            report = classify_current(s, t_seg)
            return render(s), report, s
    return (render(s) if s.gray is not None else None), "", s


S.mask_mode = MODE_RULE


def undo(s):
    if s is None or s.gray is None:
        return None, "Choose an image first.", s
    if s.source != "clicks" or not s.points:
        return render(s), "Nothing to undo.", s
    s.points.pop(); s.labels.pop(); s.mask = None; s.pred = None
    if not s.points:
        return render(s), "Click on a particle.", s
    t_seg = run_sam(s, s.mask_mode)
    if not s.mask.any():
        return render(s), "SAM returned an empty mask; try another point.", s
    report = classify_current(s, t_seg)
    return render(s), report, s


def reset(s):
    if s is None or s.gray is None:
        return None, "", s
    s.points, s.labels, s.mask, s.pred, s.source = [], [], None, None, ""
    return render(s), "Click on a particle.", s


def expert(s):
    if s is None or s.gray is None:
        return None, "Choose an image first.", s
    _, mp, _ = IMAGES[s.stem]
    if mp is None:
        return render(s), "No expert mask is shipped for this image.", s
    s.points, s.labels, s.source = [], [], "expert"
    s.mask = np.asarray(Image.open(mp).convert("L")) > 127
    report = classify_current(s)
    return render(s), report, s


with gr.Blocks(title="CNT segment + classify demo") as app:
    gr.Markdown(
        "## Segment a carbon-nanotube particle, and see it classified\n"
        "Pick a micrograph and click on a particle.  SAM proposes a mask from your click; DINOv2 "
        "features pooled **inside that mask** go to the paper's five-fold MLP ensemble, the model "
        f"whose {CLF.meta['test_acc']:.1f} % ({CLF.meta['test_correct']}/{CLF.meta['test_n']}) held-out "
        f"accuracy the paper reports.  {len(IMAGES)} held-out test images (Fiber / Cluster / Matrix), "
        "none seen in training.  Add positive points to grow a mask, negative points to trim it; "
        "*Show the expert mask* classifies the reference mask from the paper's dataset instead.")
    st = gr.State(S())
    with gr.Row():
        with gr.Column(scale=3):
            sel = gr.Dropdown(CHOICES, value=None, label="micrograph (held-out test set, fixed shuffled order)",
                              info="pick one to load it")
            img = gr.Image(type="pil", label="click on a particle", interactive=False,
                           height=int(DISPLAY_W * 1040 / 1350 * 0.75))
        with gr.Column(scale=2):
            mode = gr.Radio(["Positive point (this is the particle)",
                             "Negative point (this is background)"],
                            value="Positive point (this is the particle)", label="next click")
            mask_mode = gr.Radio([MODE_RULE, MODE_SCORE], value=MODE_RULE,
                                 label="which of SAM's three candidate masks to keep")
            with gr.Row():
                b_undo = gr.Button("undo last point"); b_reset = gr.Button("reset")
                b_expert = gr.Button("show the expert mask", variant="secondary")
            out = gr.Markdown("")
    sel.change(load, [sel, st], [img, out, st])
    img.select(click, [mode, st], [img, out, st])
    mask_mode.change(set_mask_mode, [mask_mode, st], [img, out, st])
    b_undo.click(undo, [st], [img, out, st])
    b_reset.click(reset, [st], [img, out, st])
    b_expert.click(expert, [st], [img, out, st])

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    a = ap.parse_args()
    app.launch(server_name=a.host, server_port=a.port, show_error=True, inbrowser=not a.no_browser)
