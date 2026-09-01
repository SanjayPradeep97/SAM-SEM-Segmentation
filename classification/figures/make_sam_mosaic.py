"""SAM ViT-B/16 representations of CNT morphologies -- companion to the DINOv2
mosaic (Figure 6), same four micrographs, same layout, same colour scale.

Column choice follows Ke et al. (NeurIPS 2023), who build HQ-SAM on the
observation that the feature after SAM's FIRST global-attention block "captures
more general image edge/boundary details" while "the final layer global feature
of SAM's ViT encoder has more global image context information". For SAM ViT-B
the global-attention blocks are 2, 5, 8 and 11 (global_attn_indexes=[2,5,8,11]),
so blocks 2 and 8 are the early and late global blocks.

The fourth column is the neck output -- the 256-channel, 64x64 image embedding
that is actually passed to the mask decoder. It is SAM's terminal representation
and the closest available counterpart to the DINOv2 CLS-similarity panel, which
has no SAM analogue because SAM's image encoder has no CLS token.

Maps are the L2 norm of the tokens at that stage, min-max normalised to [0,1]
and rendered with 'jet', identical to the DINOv2 recipe.
"""
import numpy as np, torch, cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # classification/
import cnt_paths
sys.path.insert(0, str(HERE))
from sam_activation import build_sam_encoder, PIXEL_MEAN, PIXEL_STD

# usage: python make_sam_mosaic.py [<output dir>]   (default <repo>/results/figures)
IMG_DIR = sys.argv[1] if len(sys.argv) > 1 else str(cnt_paths.results_dir() / "figures")
os.makedirs(IMG_DIR, exist_ok=True)
IMAGES = [(c, str(cnt_paths.image_path(c, f))) for c, f in
          [("Fiber", "CNT-Fiber-0044.tif"), ("Cluster", "CNT-Cluster-0469.tif"),
           ("Matrix", "CNT-Matrix-0461.tif"), ("MatrixSurface", "CNT-MatrixSurface-0439.tif")]]
COLS = ["Raw TEM Image", "Block 2 Activation\n(First Global Block)",
        "Block 8 Activation\n(Late Global Block)", "Neck Output\n(Image Embedding)"]

enc = build_sam_encoder()
acts = {}
for i in (2, 8):
    enc.blocks[i].register_forward_hook(
        (lambda i: (lambda m, inp, out: acts.__setitem__(i, out.detach())))(i))
enc.neck.register_forward_hook(lambda m, inp, out: acts.__setitem__("neck", out.detach()))


def norm01(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-8)


def run(path):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    small = cv2.resize(rgb, (224, 224))
    x = (cv2.resize(small, (1024, 1024)).astype(np.float64) - PIXEL_MEAN) / PIXEL_STD
    acts.clear()
    with torch.no_grad():
        enc(torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float())
    out = {i: norm01(torch.norm(acts[i][0], dim=-1).numpy()) for i in (2, 8)}
    out["neck"] = norm01(torch.norm(acts["neck"][0], dim=0).numpy())   # (C,H,W)
    return small, out


fig = plt.figure(figsize=(13.3, 12.4), dpi=300)
gs = gridspec.GridSpec(4, 4, figure=fig, wspace=0.05, hspace=0.05,
                       left=0.055, right=0.995, top=0.955, bottom=0.012)

for r, (cat, path) in enumerate(IMAGES):
    small, m = run(path)
    print(f"  {cat}: block2 {m[2].shape}, block8 {m[8].shape}, neck {m['neck'].shape}")
    panels = [small, m[2], m[8], m["neck"]]
    for c, p in enumerate(panels):
        ax = fig.add_subplot(gs[r, c])
        if c == 0:
            ax.imshow(p, cmap="gray")
        else:
            ax.imshow(cv2.resize(p, (224, 224), interpolation=cv2.INTER_LINEAR),
                      cmap="jet", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        if r == 0:
            ax.set_title(COLS[c], fontsize=12, fontweight="bold", pad=8)
        if c == 0:
            ax.text(-0.13, 0.5, cat, transform=ax.transAxes, fontsize=14,
                    fontweight="bold", va="center", ha="right", rotation=90)

out = os.path.join(IMG_DIR, "main_sam_morphology_mosaic.png")
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print("wrote", out)
