"""
test_protocol.py -- adversarial checks on paper_protocol.

v2.  An audit injected seven one-line bugs into v1 of this suite's target and
SIX PASSED ALL 20 CHECKS, including M7: early-stopping each fold model on that
fold's own evaluation set.  The tests below are built to kill each mutant by
name; the mutant IDs are cited so the mapping stays auditable.
"""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_protocol as P
from probe_fit import make_probe_fit
from sklearn.preprocessing import StandardScaler

def _find_cache():
    """Locate a cached feature file.  CNT_CACHE wins; otherwise the DINOv2
    benchmark-geometry cache for the repository's split file under
    <CNT_BASE>/Encoder Benchmark; otherwise any feats_*.npz there."""
    import glob
    import cnt_paths
    env = os.environ.get("CNT_CACHE")
    if env and os.path.exists(env):
        return env
    if cnt_paths.base(required=False) is None:
        raise SystemExit(
            "Cannot find a cached feature file: set CNT_CACHE to a feats_*.npz, or\n"
            "  CNT_BASE to the data root (caches live in <CNT_BASE>/Encoder Benchmark).\n"
            "  Build one with:  python encoder_bench.py --stage features --encoders dinov2_b14")
    bench = cnt_paths.bench_dir()
    import encoder_bench as EB
    p = bench / EB.cache_name("dinov2_b14", 32, None, 5, cnt_paths.splits_file().stem)
    if p.exists():
        return str(p)
    hits = sorted(glob.glob(str(bench / "feats_*.npz")))
    if hits:
        return hits[0]
    raise SystemExit(
        "Cannot find a cached feature file.\n"
        f"  looked under: {bench}\n"
        "  Build one with:  python encoder_bench.py --stage features --encoders dinov2_b14")


CACHE = _find_cache()
os.environ["CNT_CACHE"] = CACHE        # so the subprocess check inherits it
print(f"  cache: {CACHE}")
z = np.load(CACHE, allow_pickle=True)
y_all, split = z['y'], z['split']
dev, test = P.dev_test_indices(split)
CPU = torch.device('cpu')
QUICK = dict(epochs=12, patience=4)
quiet = lambda *a, **k: None

PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))

def load(pool="avg", head="linear"):
    X = z[f'masked|{pool}'].astype(np.float32)
    return X[dev], X[test], y_all[dev], y_all[test]

Xd, Xt, yd, yt = load()

print("\n1. SPLIT INTEGRITY")
check("dev and test disjoint", len(set(dev) & set(test)) == 0)
check("dev + test == every image", len(dev) + len(test) == len(y_all),
      f"{len(dev)} + {len(test)} = {len(y_all)}")
# The expected sizes come from the split file itself, not from a literal, so
# the suite is valid for any split (dataset_splits.pkl: 1,606 / 179).
import pickle
_raw = pickle.load(open(os.environ.get("CNT_SPLITS") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "splits",
    "dataset_splits.pkl"), "rb"))
_n_dev, _n_test = len(_raw["train_df"]) + len(_raw["val_df"]), len(_raw["test_df"])
_y_test_pkl = np.array([int(r["category_id"]) for r in _raw["test_df"]])
check(f"dev={_n_dev}, test={_n_test} as in the split file",
      len(dev) == _n_dev and len(test) == _n_test, f"cache has {len(dev)} / {len(test)}")
check("cache test labels == split-file test labels, in order",
      np.array_equal(yt, _y_test_pkl))
check("every class present in test with a 10% share",
      len(np.bincount(yt)) == 4 and (np.bincount(yt) >= 0.08 * np.bincount(y_all)).all(),
      str(np.bincount(yt)))

print("\n2. FOLDS")
f1, f2 = P.make_folds(yd, 5, 42), P.make_folds(yd, 5, 42)
check("folds reproducible", all(np.array_equal(a[0], b[0]) for a, b in zip(f1, f2)))
va_all = np.concatenate([va for _, va in f1])
check("each dev image held out exactly once",
      len(va_all) == len(yd) and len(set(va_all)) == len(yd))

print("\n3. CONTRACT ENFORCED IN run(), NOT IN THE ADAPTER   [kills M7]")
# A rogue adapter that early-stops on whatever it is asked to predict.
def rogue(train_idx, stop_idx, targets, fixed_epochs, seed):
    return [np.full((len(t) if not isinstance(t, str) else len(test), 4), .25)
            for t in targets], 1
g = P._guarded(rogue)
for nm, args in [
    ("stop_idx overlapping a prediction target",
     (np.arange(0, 100), np.arange(100, 150), [np.arange(120, 200)], None, 42)),
    ("train_idx overlapping a prediction target",
     (np.arange(0, 100), np.arange(200, 250), [np.arange(50, 150)], None, 42)),
    ("stop_idx overlapping train_idx",
     (np.arange(0, 100), np.arange(50, 150), [np.arange(200, 300)], None, 42)),
    ("early stopping with stop_idx=None", (np.arange(100), None, ["test"], None, 42)),
    ("fixed_epochs together with stop_idx",
     (np.arange(100), np.arange(200, 250), ["test"], 5, 42)),
]:
    try:
        g(*args); check(f"rejects {nm}", False, "no error raised")
    except (ValueError, TypeError):
        check(f"rejects {nm}", True)

print("\n4. ROW-ACCESS CANARY   [kills M1, M5, M7]")
class Watched(np.ndarray):
    def __getitem__(self, k):
        if isinstance(k, np.ndarray) and k.dtype != bool:
            Watched.seen.update(k.tolist())
        return super().__getitem__(k)
Watched.seen = set()
Xw = Xd.view(Watched)
i_tr, i_va = P.make_folds(yd, 5, 42)[0]
i_fit, i_es = P.inner_split(i_tr, yd, 42)
Watched.seen = set()
fit_w = make_probe_fit(Xw, yd, Xt, "linear", CPU, **QUICK)
fit_w(i_fit, i_es, [i_es], None, 42)          # train + stop only, never touch i_va
check("evaluation fold untouched while training and early-stopping",
      len(Watched.seen & set(i_va.tolist())) == 0,
      f"{len(Watched.seen & set(i_va.tolist()))} eval rows read")

print("\n5. SCALER CONTRACT   [kills M1]")
sc_ref = StandardScaler().fit(Xd[i_fit])
captured = {}
_orig_fit = StandardScaler.fit
def spy(self, X, *a, **k):
    captured['n'] = len(X); captured['mean'] = np.asarray(X).mean(0)
    return _orig_fit(self, X, *a, **k)
StandardScaler.fit = spy
make_probe_fit(Xd, yd, Xt, "linear", CPU, **QUICK)(i_fit, i_es, [i_es], None, 42)
StandardScaler.fit = _orig_fit
check("scaler fit on exactly the training rows", captured.get('n') == len(i_fit),
      f"{captured.get('n')} vs {len(i_fit)}")
check("scaler mean equals training-row mean",
      np.allclose(captured['mean'], Xd[i_fit].mean(0), atol=1e-5))

print("\n6. CORE RUN (linear + mlp)   [kills M2, M3, M4, M8]")
for head in ("linear", "mlp"):
    fit = make_probe_fit(Xd, yd, Xt, head, CPU, **QUICK)
    r = P.run(fit, yd, yt, label=head, log=quiet)
    oof_acc = 100 * (r['_oof'].argmax(1) == yd).mean()
    # Folds are not all the same size (1,606 dev images -> 322/321), so the
    # pooled out-of-fold accuracy equals the SIZE-WEIGHTED mean of the fold
    # accuracies, while cv_acc is the plain mean.  Both identities are checked.
    fw = np.array([len(va) for _, va in P.make_folds(yd, 5, 42)], dtype=float)
    check(f"[{head}] pooled OOF accuracy == size-weighted mean of fold accuracies",
          abs(oof_acc - np.average(r['cv_folds'], weights=fw)) < 1e-6,
          f"{oof_acc:.4f} vs {np.average(r['cv_folds'], weights=fw):.4f}")
    check(f"[{head}] cv_acc == mean of fold accuracies",
          abs(r['cv_acc'] - np.mean(r['cv_folds'])) < 1e-6)
    check(f"[{head}] oof_acc field matches", abs(r['oof_acc'] - oof_acc) < 1e-6)
    check(f"[{head}] every dev image has an OOF prediction", not np.isnan(r['_oof']).any())
    check(f"[{head}] epoch budget == round(median(fold epochs))",
          r['epoch_budget'] == int(round(float(np.median(r['fold_epochs'])))),
          f"{r['epoch_budget']} vs {r['fold_epochs']}")
    check(f"[{head}] ensemble probabilities normalised",
          np.allclose(r['_probs_ensemble'].sum(1), 1.0, atol=1e-4))
    check(f"[{head}] cv_std uses ddof=1",
          abs(r['cv_std'] - np.std(r['cv_folds'], ddof=1)) < 1e-6)
    check(f"[{head}] refit_unreliable follows the budget rule",
          r['refit_unreliable'] == P.refit_unreliable(r['epoch_budget']))
    if head == "linear":
        r_lin = r

print("\n7. ENSEMBLE USES ALL FIVE MEMBERS   [kills M3]")
calls = {"n": 0}
def make_tagged(perturb_member):
    base = make_probe_fit(Xd, yd, Xt, "linear", CPU, **QUICK)
    def f(train_idx, stop_idx, targets, fixed_epochs, seed):
        out, ep = base(train_idx, stop_idx, targets, fixed_epochs, seed)
        if fixed_epochs is None and stop_idx is not None and any(
                isinstance(t, str) for t in targets):
            calls["n"] += 1
            if perturb_member is not None and calls["n"] == perturb_member:
                for i, t in enumerate(targets):
                    if isinstance(t, str):
                        out[i] = np.roll(out[i], 1, axis=1)   # corrupt this member
        return out, ep
    return f

calls["n"] = 0
r_ref = P.run(make_tagged(None), yd, yt, label="ref", log=quiet, want_single=False)
moved = []
for m in range(1, 6):
    calls["n"] = 0
    r_m = P.run(make_tagged(m), yd, yt, label=f"perturb{m}", log=quiet, want_single=False)
    moved.append(not np.allclose(r_m["_probs_ensemble"], r_ref["_probs_ensemble"]))
check("perturbing ANY one of the 5 members moves the ensemble",
      all(moved), f"members that moved it: {[i+1 for i,v in enumerate(moved) if v]}")
check("ensemble equals the plain mean of its members",
      np.allclose(r_ref["_probs_ensemble"].sum(1), 1.0, atol=1e-4))

print("\n8. CHECKPOINT RESTORE   [kills M8]")
# The restored model must BE the model as it stood at its best epoch.  Training
# is deterministic given (seed, data, order), so training for exactly best_ep
# epochs with early stopping off must reproduce it bit-for-bit.  A missing
# load_state_dict() returns the LAST epoch instead, which cannot match.
fit_r = make_probe_fit(Xd, yd, Xt, "linear", CPU, epochs=60, patience=3)
(p_es,), ep_es = fit_r(i_fit, i_es, ["test"], None, 42)
(p_at_best,), _ = fit_r(i_fit, None, ["test"], ep_es, 42)
(p_last,), _ = fit_r(i_fit, None, ["test"], 60, 42)
check("early stopping actually triggered", ep_es < 60, f"best epoch {ep_es}/60")
check("restored model == model at its best epoch",
      np.allclose(p_es, p_at_best, atol=1e-5),
      f"max|diff| {np.abs(p_es - p_at_best).max():.2e}")
check("restored model != model at the final epoch",
      not np.allclose(p_es, p_last, atol=1e-6))

print("\n9. LEAK CANARIES")
rng = np.random.default_rng(0)
fit_a = make_probe_fit(Xd, yd, Xt, "linear", CPU, **QUICK)
r_a = P.run(fit_a, yd, yt, label="clean", log=quiet)
fit_b = make_probe_fit(Xd, yd, rng.normal(size=Xt.shape).astype(np.float32),
                       "linear", CPU, **QUICK)
r_b = P.run(fit_b, yd, yt, label="junk", log=quiet)
check("CV unchanged when test features are noise",
      abs(r_a['cv_acc'] - r_b['cv_acc']) < 1e-9)
check("test collapses on noise test set", r_b['test_ensemble_acc'] < 45,
      f"{r_b['test_ensemble_acc']:.2f}%")
yd_s = rng.permutation(yd)
r_d = P.run(make_probe_fit(Xd, yd_s, Xt, "linear", CPU, **QUICK), yd_s, yt,
            label="shuf", log=quiet)
check("CV collapses when dev labels are shuffled", r_d['cv_acc'] < 40,
      f"{r_d['cv_acc']:.2f}%")
check("fit never receives y_test",
      'y_test' not in make_probe_fit.__code__.co_varnames)

print("\n10. DETERMINISM ACROSS PROCESSES")
import subprocess
code = ('import sys,numpy as np,torch;sys.path.insert(0,%r);'
        'import paper_protocol as P;from probe_fit import make_probe_fit;'
        'z=np.load(%r,allow_pickle=True);y=z["y"];s=z["split"];'
        'd,t=P.dev_test_indices(s);X=z["masked|avg"].astype(np.float32);'
        'f=make_probe_fit(X[d],y[d],X[t],"linear",torch.device("cpu"),epochs=12,patience=4);'
        'r=P.run(f,y[d],y[t],log=lambda *a:None);'
        'print(round(r["cv_acc"],9),round(r["test_ensemble_acc"],9))'
        % (os.path.dirname(os.path.abspath(__file__)), CACHE))
env = dict(os.environ, OMP_NUM_THREADS="1")
o = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
got = o.stdout.strip().split()
check("identical across a fresh process with OMP_NUM_THREADS=1",
      len(got) == 2 and abs(float(got[0]) - r_a['cv_acc']) < 1e-6,
      f"{got} vs {r_a['cv_acc']:.6f}")

print("\n11. NO BARE ASSERTS IN THE PROTOCOL")
srcp = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'paper_protocol.py')).read()
check("no bare assert statements (they vanish under python -O)",
      not any(l.strip().startswith("assert ") for l in srcp.splitlines()))

print(f"\n{'='*64}\n  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL)); sys.exit(1)
print("  all protocol guarantees hold")
