"""Does the periodic structure in SAM's activation maps come from its 14x14
window tiling, or from the positional embedding?

The published explanation for grid artifacts in ViT feature maps is the
positional embedding (Yang et al., ECCV 2024), and SAM has a learned absolute
positional embedding, so the window explanation cannot simply be assumed. Three
tests discriminate between them:

  T1  PERIOD AND ASYMMETRY.  SAM ViT-B tiles a 64x64 token grid with 14x14
      non-overlapping windows, zero-padded to 70x70 -> 5x5 windows. Boundaries
      therefore fall at token indices 14, 28, 42, 56 and the final band is only
      8 real tokens wide. A period-14 structure WITH a short final band is a
      signature of this tiling; a positional embedding has no reason to produce
      an asymmetric last band.

  T2  WINDOWED vs GLOBAL BLOCKS.  Blocks 2, 5, 8, 11 use global attention; the
      other eight are windowed. Periodicity should be weaker at global blocks.

  T3  ABLATE THE POSITIONAL EMBEDDING.  Zero pos_embed and re-measure. If the
      periodicity survives, the positional-embedding account is ruled out.
"""
import numpy as np, torch, cv2
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # classification/
import cnt_paths
sys.path.insert(0, str(HERE))
from sam_activation import build_sam_encoder, PIXEL_MEAN, PIXEL_STD
from PIL import Image

SRC = cnt_paths.image_path("Fiber", "CNT-Fiber-0044.tif")
GLOBAL_BLOCKS = {2, 5, 8, 11}


def maps(enc, image_rgb, layers):
    acts = {}
    hs = [enc.blocks[i].register_forward_hook(
        (lambda i: (lambda m, inp, out: acts.__setitem__(i, out.detach())))(i)) for i in layers]
    img = cv2.resize(cv2.resize(image_rgb, (224, 224)), (1024, 1024))
    x = (img.astype(np.float64) - PIXEL_MEAN) / PIXEL_STD
    with torch.no_grad():
        enc(torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float())
    for h in hs: h.remove()
    return {i: torch.norm(a[0], dim=-1).cpu().numpy() for i, a in acts.items()}


def boundary_contrast(g):
    """Mean |difference| across the four window boundaries, divided by the mean
    |difference| across all other interior columns/rows. >1 means the boundaries
    are more discontinuous than a typical neighbouring pair."""
    d_col = np.abs(np.diff(g, axis=1)); d_row = np.abs(np.diff(g, axis=0))
    bnd = [13, 27, 41, 55]                      # diff index j spans columns j, j+1
    oth = [j for j in range(g.shape[1] - 1) if j not in bnd]
    return ((d_col[:, bnd].mean() + d_row[bnd, :].mean()) /
            (d_col[:, oth].mean() + d_row[oth, :].mean()))


img = np.asarray(Image.open(SRC).convert("RGB"))
enc = build_sam_encoder()
layers = list(range(12))
M = maps(enc, img, layers)

print("T1/T2  boundary contrast at the four window seams (>1 = seam visible)")
print(f"  {'block':>6s} {'attention':>10s} {'contrast':>9s}")
w, gl = [], []
for i in layers:
    c = boundary_contrast(M[i])
    kind = "GLOBAL" if i in GLOBAL_BLOCKS else "windowed"
    (gl if i in GLOBAL_BLOCKS else w).append(c)
    print(f"  {i:6d} {kind:>10s} {c:9.3f}")
print(f"\n  mean windowed {np.mean(w):.3f}   mean global {np.mean(gl):.3f}   "
      f"ratio {np.mean(w)/np.mean(gl):.2f}x")

print("\nT1  column-profile periodicity (mean activation per token column)")
prof = M[10].mean(0)
seam = prof[[13, 14, 27, 28, 41, 42, 55, 56]].mean()
rest = np.delete(prof, [13, 14, 27, 28, 41, 42, 55, 56]).mean()
print(f"  mean at seam columns {seam:.3f}   elsewhere {rest:.3f}   ratio {seam/rest:.3f}")
print(f"  final band is columns 56..63 = {64-56} tokens wide (a full window is 14)")

print("\nT3  zero the absolute positional embedding and re-measure")
with torch.no_grad():
    enc.pos_embed.zero_()
M0 = maps(enc, img, layers)
w0 = [boundary_contrast(M0[i]) for i in layers if i not in GLOBAL_BLOCKS]
g0 = [boundary_contrast(M0[i]) for i in layers if i in GLOBAL_BLOCKS]
print(f"  mean windowed {np.mean(w0):.3f}   mean global {np.mean(g0):.3f}   "
      f"ratio {np.mean(w0)/np.mean(g0):.2f}x")
print(f"  windowed-block contrast retained without pos_embed: "
      f"{100*np.mean(w0)/np.mean(w):.0f}% of the original")
