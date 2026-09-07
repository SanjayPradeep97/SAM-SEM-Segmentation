"""Train the classifier heads shipped with the demo, from the feature cache.

What ships is the paper's reported model: the FIVE fold heads of the
pre-specified configuration (paper_protocol.PRIMARY), averaged as a softmax
ensemble.  That is exactly the `test_ensemble` variant whose 92.7 % (166/179)
the manuscript quotes.  Each fold head is trained by probe_fit.make_probe_fit,
the same code the benchmark used, on that fold's 90 % training split with
early stopping on its 10 % inner split; the held-out fold and the test set
play no part in training or stopping.

This script refuses to save unless the ensemble's test-set predictions are
identical, image for image, to the ones recorded in results/per_image_probe.json
for the PRIMARY row.  So the shipped weights cannot silently be a different
model from the one the paper reports.

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
ap.add_argument("--out", type=Path, default=HERE / "weights" / "dinov2_probe_heads.pt")
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
KEY = f"{enc}[{geom[0]}]|{P.PRIMARY['mask']}|{P.PRIMARY['pooling']}|{P.PRIMARY['classifier']}"
rec = next(e for e in json.load(open(cnt_paths.results_dir() / "per_image_probe.json")) if e["key"] == KEY)
ref_pred = np.array(rec["pred_ensemble"])
if not np.array_equal(np.array(rec["y_test"]), yt):
    raise SystemExit("test labels in per_image_probe.json differ from the cache")
row = next(r for r in csv.DictReader(open(cnt_paths.results_dir() / "results_probes.csv"))
           if (r["method"], r["geometry"], r["mask"], r["pooling"], r["classifier"]) ==
              (enc, geom[0], P.PRIMARY["mask"], P.PRIMARY["pooling"], P.PRIMARY["classifier"]))
SEED, N_FOLDS = 42, 5
print(f"cache   {cache.name}   dev {Xd.shape}  test {Xt.shape}   device {dev}")
print(f"heads   {N_FOLDS} fold models of {P.PRIMARY['classifier']}, seed {SEED}, early-stopped on inner splits")


def train_one(train_idx, stop_idx, seed, epochs=100, patience=10, batch=64):
    """probe_fit.make_probe_fit's early-stopped training, keeping the model."""
    sc = StandardScaler().fit(Xd[train_idx])
    Xtr = torch.tensor(sc.transform(Xd[train_idx]), device=dev)
    ytr = torch.tensor(yd[train_idx], device=dev, dtype=torch.long)
    Xes = torch.tensor(sc.transform(Xd[stop_idx]), device=dev)
    yes = torch.tensor(yd[stop_idx], device=dev, dtype=torch.long)
    model = PF.build_head(Xd.shape[1], P.PRIMARY["classifier"], seed).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = torch.nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(seed)
    n = len(train_idx); best, wait, best_state, best_ep = float("inf"), 0, None, epochs
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            if len(idx) < 2:
                continue
            opt.zero_grad(); crit(model(Xtr[idx]), ytr[idx]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xes), yes).item()
        if vl < best - 1e-5:
            best, wait, best_ep = vl, 0, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state); model.eval()
    return model, sc, best_ep


t0 = time.time()
folds = P.make_folds(yd, N_FOLDS, SEED)
heads, fold_epochs, fold_acc, test_probs = [], [], [], []
fit_ref = PF.make_probe_fit(Xd, yd, Xt, P.PRIMARY["classifier"], dev)
for k, (i_tr, i_va) in enumerate(folds, 1):
    i_fit, i_es = P.inner_split(i_tr, yd, SEED)
    model, sc, ep = train_one(i_fit, i_es, SEED)
    with torch.no_grad():
        p_va = torch.softmax(model(torch.tensor(sc.transform(Xd[i_va]), device=dev)), 1).cpu().numpy()
        p_te = torch.softmax(model(torch.tensor(sc.transform(Xt), device=dev)), 1).cpu().numpy()
    # cross-check this fold against the protocol's own closure (same inputs, same seed)
    (r_va, r_te), r_ep = fit_ref(i_fit, i_es, [i_va, "test"], None, SEED)
    if not (np.allclose(p_te, r_te, atol=1e-4) and r_ep == ep):
        raise SystemExit(f"fold {k}: kept model differs from probe_fit's (epoch {ep} vs {r_ep})")
    acc = 100 * (p_va.argmax(1) == yd[i_va]).mean()
    fold_epochs.append(ep); fold_acc.append(acc); test_probs.append(p_te)
    heads.append(dict(state_dict={kk: v.detach().cpu().half() for kk, v in model.state_dict().items()},
                      scaler_mean=torch.from_numpy(sc.mean_.astype(np.float32)),
                      scaler_scale=torch.from_numpy(sc.scale_.astype(np.float32)),
                      best_epoch=int(ep)))
    print(f"  fold {k}/{N_FOLDS}: {acc:5.2f}% on its held-out fold  (best epoch {ep})")

ens = np.mean(test_probs, axis=0)
pred = ens.argmax(1)
k_ok = int((pred == yt).sum()); acc = 100.0 * k_ok / len(yt)
lo, hi = P.wilson_ci(k_ok, len(yt))
print(f"\nensemble on the held-out test set: {acc:.2f}%  ({k_ok}/{len(yt)})  95% CI [{lo:.1f}, {hi:.1f}]"
      f"   ({time.time()-t0:.0f}s)")
print(f"results_probes.csv test_ensemble_acc for this row: {float(row['test_ensemble_acc']):.2f}%  "
      f"fold epochs {row['fold_epochs']}")
same = np.array_equal(pred, ref_pred)
print(f"test predictions identical to per_image_probe.json, image for image: {same}")
if not same or json.loads(row["fold_epochs"]) != fold_epochs:
    raise SystemExit("the shipped heads would not be the paper's model; refusing to save")

a.out.parent.mkdir(parents=True, exist_ok=True)
torch.save(dict(
    heads=heads,
    meta=dict(encoder=enc, geometry=geom[0], target_grid=geom[1], taps=list(geom[2]),
              img_size=meta_cache["img_size"], mask=P.PRIMARY["mask"],
              pooling=P.PRIMARY["pooling"], classifier=P.PRIMARY["classifier"],
              variant="test_ensemble", n_folds=N_FOLDS, seed=SEED, fold_epochs=fold_epochs,
              fold_acc=[float(x) for x in fold_acc], n_dev=int(len(yd)),
              split_file=cnt_paths.splits_file().name, device=str(dev),
              test_acc=float(acc), test_correct=k_ok, test_n=int(len(yt)),
              test_ci=[float(lo), float(hi)], classes=["Fiber", "Cluster", "Matrix", "MatrixSurface"],
              results_key=KEY,
              note=(f"the five fold heads of the pre-specified configuration, averaged as a "
                    f"softmax ensemble (protocol variant 'test_ensemble'): the model whose "
                    f"{acc:.1f}% ({k_ok}/{len(yt)}) the paper reports"))),
    a.out)
print(f"wrote {a.out}  ({a.out.stat().st_size/1e6:.1f} MB)")
old = HERE / "weights" / "dinov2_probe_head.pt"
if old.exists():
    old.unlink(); print(f"removed the superseded single-head file {old.name}")
