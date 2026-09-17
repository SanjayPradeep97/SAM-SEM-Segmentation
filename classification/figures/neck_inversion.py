"""Is SAM's terminal image embedding higher on the particle or on the background?

The mosaic shows the neck output inverted relative to the block activations --
the particle appears LOW and the background HIGH. This measures it against the
ground-truth masks so the claim rests on a number rather than on the colour map.
"""
import numpy as np, torch, cv2
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # classification/
import cnt_paths
sys.path.insert(0, str(HERE))
from sam_activation import build_sam_encoder, PIXEL_MEAN, PIXEL_STD
from PIL import Image

# The four micrographs of Figures 6 and 7.  Prints the in-mask / background
# token-norm ratios quoted in the Figure 7 caption (1.05 at block 8, 0.97 at
# the neck).
ITEMS = [("Fiber", "CNT-Fiber-0044"), ("Cluster", "CNT-Cluster-0469"),
         ("Matrix", "CNT-Matrix-0461"), ("MatrixSurface", "CNT-MatrixSurface-0439")]
enc = build_sam_encoder(); acts = {}
for i in (2, 8):
    enc.blocks[i].register_forward_hook(
        (lambda i: (lambda m,inp,out: acts.__setitem__(i, out.detach())))(i))
enc.neck.register_forward_hook(lambda m,inp,out: acts.__setitem__("neck", out.detach()))

print(f"  {'image':16s} {'stage':>10s} {'in-mask':>9s} {'background':>11s} {'ratio':>7s}")
agg = {}
for cat, stem in ITEMS:
    rgb = np.asarray(Image.open(cnt_paths.image_path(cat, f"{stem}.tif")).convert("RGB"))
    mp = cnt_paths.mask_path(f"{stem}.tif")
    m = np.asarray(Image.open(mp).convert("L")) > 127
    m64 = cv2.resize(m.astype(np.uint8), (64,64), interpolation=cv2.INTER_AREA) > 0.5
    x = (cv2.resize(cv2.resize(rgb,(224,224)),(1024,1024)).astype(np.float64)-PIXEL_MEAN)/PIXEL_STD
    acts.clear()
    with torch.no_grad(): enc(torch.from_numpy(x).permute(2,0,1).unsqueeze(0).float())
    g = {2: torch.norm(acts[2][0],dim=-1).numpy(), 8: torch.norm(acts[8][0],dim=-1).numpy(),
         "neck": torch.norm(acts["neck"][0],dim=0).numpy()}
    for k in (2,8,"neck"):
        a=g[k][m64].mean(); b=g[k][~m64].mean()
        agg.setdefault(k,[]).append(a/b)
        print(f"  {cat:16s} {str(k):>10s} {a:9.2f} {b:11.2f} {a/b:7.3f}")
print()
for k in (2,8,"neck"):
    v=np.array(agg[k])
    print(f"  {str(k):>6s}: mean in-mask/background ratio {v.mean():.3f}  "
          f"({'HIGHER on particle' if v.mean()>1 else 'HIGHER on background'}, all four: "
          f"{'yes' if (v>1).all() or (v<1).all() else 'mixed'})")
