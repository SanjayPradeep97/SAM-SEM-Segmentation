"""classify.py -- the paper's classifier, applied to ONE image and ONE mask.

This is the inference half of the pipeline the manuscript describes: a frozen
DINOv2 ViT-B/14 at 518x518 with hypercolumn taps at blocks [1, 3, 6, 9, 11],
mask-guided average+max pooling (7,680-D), a StandardScaler fitted on each
fold's training rows, and the two-layer MLP head.  Feature extraction calls
the SAME functions the benchmark used (encoder_bench.build_encoder /
pool_maps), so what the demo computes is what the reported numbers were
computed from.

The weights in weights/dinov2_probe_heads.pt are the five fold heads of the
pre-specified configuration, produced by train_demo_head.py through the same
probe_fit code as the benchmark and averaged as a softmax ensemble.  That is
the `test_ensemble` variant the paper reports (92.7 %, 166/179 on the held-out
test set); train_demo_head.py refuses to save heads whose test predictions
differ from results/per_image_probe.json by even one image.

    from classify import CNTClassifier
    clf = CNTClassifier()                      # loads DINOv2 (timm) + the heads
    label, probs = clf.predict(gray_uint8_HxW, mask_bool_HxW)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "classification"))

CLASSES = ["Fiber", "Cluster", "Matrix", "MatrixSurface"]
ENCODER = "dinov2_b14"
TARGET_GRID = 37                 # 37 x 14 px = 518 px input
TAPS = [1, 3, 6, 9, 11]
WEIGHTS = HERE / "weights" / "dinov2_probe_heads.pt"


class CNTClassifier:
    def __init__(self, weights: Path = WEIGHTS, device=None, verbose=True):
        import torch
        import encoder_bench as EB
        import probe_fit as PF
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if not Path(weights).is_file():
            raise FileNotFoundError(
                f"{weights} not found.  Regenerate it with demo/train_demo_head.py "
                f"(needs the DINOv2 feature cache under CNT_BASE).")
        ck = torch.load(weights, map_location="cpu")
        self.meta = ck["meta"]
        if self.meta["encoder"] != ENCODER or self.meta["taps"] != TAPS \
           or self.meta["target_grid"] != TARGET_GRID:
            raise ValueError(f"heads were trained for {self.meta}, not for this extractor")
        self.heads = []
        for h in ck["heads"]:
            mean = h["scaler_mean"].numpy().astype(np.float32)
            scale = h["scaler_scale"].numpy().astype(np.float32)
            net = PF.build_head(len(mean), self.meta["classifier"], 0)
            net.load_state_dict({k: v.float() for k, v in h["state_dict"].items()})
            net.to(self.device).eval()
            self.heads.append((net, mean, scale))
        self.dim_head = len(self.heads[0][1])

        (self.enc, self.enc_mean, self.enc_std, self.img_size, n_avail,
         grids, chans, _, _) = EB.build_encoder(ENCODER, TARGET_GRID, 512, self.device)
        if any(t >= n_avail for t in TAPS):
            raise RuntimeError(f"encoder exposes {n_avail} blocks; taps {TAPS} out of range")
        self.dim = 2 * sum(chans[t] for t in TAPS)
        if self.dim != self.dim_head:
            raise RuntimeError(f"feature dim {self.dim} != head input {self.dim_head}")
        if verbose:
            m = self.meta
            print(f"[classify] {ENCODER} @ {self.img_size}px, taps {TAPS}, "
                  f"{self.dim}-D masked avg+max -> {len(self.heads)}-fold {m['classifier']} ensemble")
            print(f"[classify] {m['note']}")
            print(f"[classify] held-out test accuracy: {m['test_acc']:.2f}% "
                  f"({m['test_correct']}/{m['test_n']})   device {self.device}")

    # ------------------------------------------------------------------
    def features(self, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """7,680-D masked avg+max hypercolumn, exactly as encoder_bench.extract()."""
        import torch
        from PIL import Image
        import encoder_bench as EB
        if gray.ndim == 3:
            gray = np.asarray(Image.fromarray(gray).convert("L"))
        im = Image.fromarray(gray.astype(np.uint8), mode="L")
        mk = Image.fromarray((np.asarray(mask).astype(bool) * 255).astype(np.uint8), mode="L")
        s = self.img_size
        x = np.asarray(im.resize((s, s), Image.BILINEAR), dtype=np.float32) / 255.0
        m = np.asarray(mk.resize((s, s), Image.BILINEAR), dtype=np.float32) / 255.0
        with torch.no_grad():
            xt = torch.from_numpy(x).to(self.device)[None, None].repeat(1, 3, 1, 1)
            xt = (xt - self.enc_mean) / self.enc_std
            mt = torch.from_numpy(m).to(self.device)[None, None]
            with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                enabled=self.device.type == "cuda"):
                feats = self.enc.forward_intermediates(xt, indices=TAPS,
                                                       intermediates_only=True,
                                                       output_fmt="NCHW")
            pooled = EB.pool_maps(feats, mt, list(range(len(feats))))
        return pooled[("masked", "avg+max")].float().cpu().numpy()[0]

    def predict_features(self, f: np.ndarray):
        """Mean softmax over the five fold heads -- the paper's ensemble rule."""
        import torch
        probs = np.zeros(len(CLASSES), np.float64)
        with torch.no_grad():
            for net, mean, scale in self.heads:
                z = (f.astype(np.float32) - mean) / scale
                probs += torch.softmax(net(torch.from_numpy(z)[None].to(self.device)), 1)[0].cpu().numpy()
        probs /= len(self.heads)
        return CLASSES[int(probs.argmax())], probs

    def predict(self, gray: np.ndarray, mask: np.ndarray):
        """Returns (label, probs[4]) for one particle mask on one micrograph."""
        if not np.asarray(mask).any():
            raise ValueError("empty mask: nothing to classify")
        return self.predict_features(self.features(gray, mask))


if __name__ == "__main__":
    import argparse
    from PIL import Image
    ap = argparse.ArgumentParser(description="classify one particle: image + mask -> label")
    ap.add_argument("image"); ap.add_argument("mask")
    a = ap.parse_args()
    clf = CNTClassifier()
    g = np.asarray(Image.open(a.image).convert("L"))
    m = np.asarray(Image.open(a.mask).convert("L")) > 127
    lab, pr = clf.predict(g, m)
    print(lab, {c: round(float(p), 3) for c, p in zip(CLASSES, pr)})
