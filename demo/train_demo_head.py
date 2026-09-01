"""Train the classifier head shipped with the demo, from the feature cache.

The head is the `test_refit` variant of the pre-specified configuration
(paper_protocol.PRIMARY): ONE model fitted on all development images, for
exactly the epoch budget the five folds transferred (results_probes.csv,
column `epoch_budget`), with early stopping OFF -- so the held-out test set
plays no part in training or stopping.  The training loop is
probe_fit.make_probe_fit, i.e. the same code the reported numbers come from;
this script only additionally keeps the trained weights.

The test accuracy printed at the end is scored once and written into the
weights file so classify.py can display it.  It is expected to equal
`test_refit_acc` of the PRIMARY row in results_probes.csv when run on the
same device class the results were produced on (GPU); CPU arithmetic can
differ in the last epoch by a prediction or two.

usage:  set CNT_BASE=<data root>    (the feature cache lives there)
        python train_demo_head.py [--device cuda|cpu]
"""
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "classification"))
import cnt_paths
import paper_protocol as P
import paper_results as PR
import probe_fit as PF

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="auto")
ap.add_argument("--out", type=Path, default=HERE / "weights" / "dinov2_probe_head.pt")
a = ap.parse_args()

import torch
from sklearn.preprocessing import StandardScaler
dev = torch.device("cuda" if (a.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")

enc, geom = P.PRIMARY["encoder"], PR.GEOM_PAPER
cache = PR.cache_path(enc, geom)
if not cache.exists():
    raise SystemExit(f"feature cache missing: {cache}\n  build it with:\n"
                     f"  python classification/encoder_bench.py --stage features "
                     f"--encoders dinov2_b14 --target-grid 37 --taps 1,3,6,9,11")
z = np.load(cache, allow_pickle=True)
y_all, split = z["y"], z["split"]
di, ti = P.dev_test_indices(split)
X = z[f"{P.PRIMARY['mask']}|{P.PRIMARY['pooling']}"].astype(np.float32)
Xd, Xt, yd, yt = X[di], X[ti], y_all[di], y_all[ti]
meta_cache = json.loads(str(z["meta"]))

# the transferred epoch budget, read from the results file rather than re-derived
prim = next(r for r in csv.DictReader(open(cnt_paths.results_dir() / "results_probes.csv"))
            if (r["method"], r["geometry"], r["mask"], r["pooling"], r["classifier"]) ==
               (enc, geom[0], P.PRIMARY["mask"], P.PRIMARY["pooling"], P.PRIMARY["classifier"]))
budget, seed = int(prim["epoch_budget"]), 42
print(f"cache   {cache.name}   dev {Xd.shape}  test {Xt.shape}")
print(f"head    {P.PRIMARY['classifier']}  refit on all {len(yd)} dev images for {budget} epochs "
      f"(fold epochs {prim['fold_epochs']}), seed {seed}, device {dev}")

# --- identical to probe_fit.make_probe_fit(...)(all_dev, None, ["test"], budget, seed),
#     except that the trained model is kept.  Kept in step by asserting the
#     probabilities agree with the protocol's own fit closure below.
sc = StandardScaler().fit(Xd)
Xtr = torch.tensor(sc.transform(Xd), device=dev)
ytr = torch.tensor(yd, device=dev, dtype=torch.long)
model = PF.build_head(Xd.shape[1], P.PRIMARY["classifier"], seed).to(dev)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
crit = torch.nn.CrossEntropyLoss()
g = torch.Generator().manual_seed(seed)
n, batch = len(yd), 64
t0 = time.time()
for ep in range(1, budget + 1):
    model.train()
    perm = torch.randperm(n, generator=g).to(dev)
    for i in range(0, n, batch):
        idx = perm[i:i + batch]
        if len(idx) < 2:
            continue
        opt.zero_grad(); crit(model(Xtr[idx]), ytr[idx]).backward(); opt.step()
model.eval()
with torch.no_grad():
    probs = torch.softmax(model(torch.tensor(sc.transform(Xt), device=dev)), 1).cpu().numpy()

# cross-check against the protocol's own closure (same seed, same budget)
fit = PF.make_probe_fit(Xd, yd, Xt, P.PRIMARY["classifier"], dev)
(p_ref,), _ = fit(np.arange(len(yd)), None, ["test"], budget, seed)
agree = np.allclose(probs, p_ref, atol=1e-4)
pred = probs.argmax(1)
acc = 100.0 * (pred == yt).mean(); k = int((pred == yt).sum())
lo, hi = P.wilson_ci(k, len(yt))
print(f"trained in {time.time()-t0:.1f}s;  agrees with probe_fit closure: {agree}")
print(f"held-out test accuracy (scored once): {acc:.2f}%  ({k}/{len(yt)})  95% CI [{lo:.1f}, {hi:.1f}]")
print(f"results_probes.csv test_refit_acc for this row: {float(prim['test_refit_acc']):.2f}%")
if not agree:
    raise SystemExit("the kept model does not reproduce the protocol's refit; refusing to save")

a.out.parent.mkdir(parents=True, exist_ok=True)
torch.save(dict(
    state_dict={k: v.detach().cpu().half() for k, v in model.state_dict().items()},
    scaler_mean=torch.from_numpy(sc.mean_.astype(np.float32)),
    scaler_scale=torch.from_numpy(sc.scale_.astype(np.float32)),
    meta=dict(encoder=enc, geometry=geom[0], target_grid=geom[1], taps=list(geom[2]),
              img_size=meta_cache["img_size"], mask=P.PRIMARY["mask"],
              pooling=P.PRIMARY["pooling"], classifier=P.PRIMARY["classifier"],
              variant="test_refit", epochs=budget, seed=seed, n_train=int(len(yd)),
              split_file=cnt_paths.splits_file().name, device=str(dev),
              test_acc=float(acc), test_correct=k, test_n=int(len(yt)),
              test_ci=[float(lo), float(hi)], classes=["Fiber", "Cluster", "Matrix", "MatrixSurface"],
              note=(f"single refit of the pre-specified configuration on all {len(yd)} "
                    f"development images for {budget} epochs (protocol variant "
                    f"'test_refit'); NOT the five-fold ensemble the paper reports"))),
    a.out)
print(f"wrote {a.out}  ({a.out.stat().st_size/1e6:.1f} MB)")
