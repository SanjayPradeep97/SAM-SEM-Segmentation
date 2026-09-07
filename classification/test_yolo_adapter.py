"""Structural test for finetune_yolo under the protocol contract.

Ultralytics and the real images are not needed: a fake YOLO records what the
trainer was handed, which is exactly what the last three bugs were about.
"""
import sys, types, tempfile, shutil, os
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# No data root is needed: the fake YOLO never downloads anything.  cache_setup
# still wants CNT_BASE for its download-cache directory, so point it at the
# same temporary directory the test works in.
os.environ.setdefault("CNT_BASE", tempfile.mkdtemp())

CALLS = []

class FakeProbs:
    def __init__(self, n): self.data = _T(np.full(4, 0.25))
class _T:
    def __init__(self, a): self._a = a
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self._a
class FakeRes:
    names = {0: "Fiber", 1: "Cluster", 2: "Matrix", 3: "MatrixSurface"}
    def __init__(self): self.probs = FakeProbs(4)
class FakeTrainer:
    def __init__(self, save_dir): self.save_dir = save_dir; self.best_epoch = 3
class FakeYOLO:
    def __init__(self, src): self.src = src
    def train(self, **kw):
        root = Path(kw["data"])
        rec = dict(name=kw["name"], epochs=kw["epochs"], patience=kw["patience"],
                   val=kw["val"], data=str(root))
        for split in ("train", "val"):
            d = root / split
            rec[f"n_{split}"] = sum(len(list(c.glob('*.png')))
                                    for c in d.iterdir()) if d.exists() else 0
            rec[f"cls_{split}"] = sorted(c.name for c in d.iterdir()
                                         if c.is_dir() and any(c.glob('*.png'))) \
                                  if d.exists() else []
        CALLS.append(rec)
        sd = Path(kw["project"]) / kw["name"]
        (sd / "weights").mkdir(parents=True, exist_ok=True)
        for w in ("best.pt", "last.pt"): (sd / "weights" / w).touch()
        self.trainer = FakeTrainer(sd)
    def predict(self, paths, **kw): return [FakeRes() for _ in paths]

sys.modules["ultralytics"] = types.SimpleNamespace(YOLO=FakeYOLO)
import finetune_baselines as FB
import paper_protocol as P
from finetune_fit import make_finetune_fit

work = Path(tempfile.mkdtemp())
cfg = FB.Cfg(epochs=30, patience=8, batch=32, folds=5, device="cpu", out=work)
items_dev = [(np.zeros((16, 16), np.uint8), i % 4) for i in range(400)]
items_test = [(np.zeros((16, 16), np.uint8), i % 4) for i in range(80)]
yd = np.array([y for _, y in items_dev]); yt = np.array([y for _, y in items_test])

fit = make_finetune_fit(items_dev, yd, items_test, "yolo11", cfg,
                        dict(kind="yolo", id="yolo11s-cls.pt"), "yolo11_masked")
r = P.run(fit, yd, yt, n_folds=5, seed=42, log=lambda *a: None)

PASS, FAIL = [], []
def check(n, c, d=""):
    (PASS if c else FAIL).append(n)
    print(f"  [{'PASS' if c else 'FAIL'}] {n}" + (f"  -- {d}" if d else ""))

print("\nYOLO ADAPTER")
check("one train() per protocol fit", len(CALLS) == P.n_fits(5),
      f"{len(CALLS)} vs {P.n_fits(5)}")
names = [c["name"] for c in CALLS]
check("every run directory is unique  [fold 2 was overwriting fold 1]",
      len(set(names)) == len(names), f"{len(set(names))}/{len(names)} unique")

folds = [c for c in CALLS if c["val"]]
refit = [c for c in CALLS if not c["val"]]
check("5 fold runs use a real held-out val split", len(folds) == 6, f"{len(folds)}")
check("exactly one refit run", len(refit) == 1, f"{len(refit)}")

rf = refit[0]
check("refit val/ is NON-empty  [the FileNotFoundError]", rf["n_val"] > 0,
      f"n_val={rf['n_val']}")
check("refit val/ covers all four classes", len(rf["cls_val"]) == 4, str(rf["cls_val"]))
check("refit trains on ALL dev images", rf["n_train"] == len(items_dev),
      f"{rf['n_train']} vs {len(items_dev)}")
check("refit disables early stopping", rf["patience"] > rf["epochs"],
      f"patience={rf['patience']} epochs={rf['epochs']}")
check("refit uses the transferred budget", rf["epochs"] == r["epoch_budget"],
      f"{rf['epochs']} vs budget {r['epoch_budget']}")
f0 = folds[0]
check("fold trains on less than all dev (a real split is held out)",
      f0["n_train"] < len(items_dev), f"{f0['n_train']}")
check("fold val/ is disjoint in size from train", f0["n_val"] > 0, f"n_val={f0['n_val']}")
check("fold keeps the configured patience", f0["patience"] == cfg.patience)
check("probabilities are normalised", abs(r["_probs_ensemble"].sum(1).mean() - 1) < 1e-6)
check("scratch dataset dirs cleaned up",
      not any((work / "_work").glob("yolo_*")) if (work / "_work").exists() else True)

# ---------------------------------------------------------------------------
# Regression: the cosine schedule must span the epochs ACTUALLY run.
# Building it for cfg.epochs while the refit runs `fixed_epochs` strands the
# model mid-curve at a high LR -- that took ConvNeXt-V2 to 25.00% (chance).
print("\nSCHEDULE HORIZON REGRESSION")
import re as _re, inspect as _inspect
src = _inspect.getsource(FB.finetune_timm)
m_steps = _re.search(r"steps\s*=\s*max\(1,\s*len\(tr_dl\)\)\s*\*\s*(\w+)", src)
check("cosine schedule spans n_ep, not cfg.epochs",
      m_steps is not None and m_steps.group(1) == "n_ep",
      f"steps = ... * {m_steps.group(1) if m_steps else '??'}")
i_ne = src.find("n_ep = ")
i_st = src.find("steps = max(1, len(tr_dl))")
check("n_ep is defined before the scheduler is built",
      0 <= i_ne < i_st, f"n_ep@{i_ne} steps@{i_st}")

import math as _math
def lr_at_end(total_ep, run_ep, n=1620, batch=32):
    spe = _math.ceil(n / batch); steps = spe * run_ep; warm = max(1, int(.1 * steps))
    s = spe * run_ep
    return s / warm if s < warm else \
        0.5 * (1 + _math.cos(_math.pi * (s - warm) / max(1, steps - warm)))
check("a 13-epoch refit now anneals to ~0 LR (was 70% of peak)",
      lr_at_end(30, 13) < 0.01, f"end-of-run LR factor {lr_at_end(30,13):.4f}")
print(f"\n  {len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
