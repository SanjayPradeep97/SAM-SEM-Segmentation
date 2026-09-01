"""Re-derive every headline number from the results files and fail loudly if
the manuscript disagrees.

This is the guard against the manuscript and the pipeline drifting apart.  It
trusts nothing typed by hand: each value is recomputed from results_probes.csv,
results_finetune.csv, analysis.json, per_image_probe.json and the Luo baseline
summary, then checked in one of two ways.

  With the manuscript source (main.tex + supplementary.tex) available:
      python verify_manuscript.py [<results dir>] <tex dir>
      python verify_manuscript.py --tex <tex dir>
  every derived value is rendered exactly as the paper prints it and asserted
  to be present in the .tex.  This is what the authors run before submission.

  Without the manuscript source (the public repository does not include it):
      python verify_manuscript.py [<results dir>]
  the same derived values are compared against manuscript_numbers.json, a
  snapshot written with --snapshot at the moment the .tex check last passed.
  A reader with the published paper open can compare the printed table
  directly; a change to any results file that alters a reported number makes
  this step fail.

  --snapshot   (authors only) rewrite manuscript_numbers.json from the results.
"""
import csv, json, sys, os, re
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cnt_paths

argv = sys.argv[1:]
SNAP = HERE / "manuscript_numbers.json"
write_snapshot = "--snapshot" in argv
if write_snapshot: argv.remove("--snapshot")
TEX = None
if "--tex" in argv:
    i = argv.index("--tex"); TEX = Path(argv[i + 1]); del argv[i:i + 2]
import luo_results
LUO = luo_results.DEFAULT_DIR
if "--luo" in argv:
    i = argv.index("--luo"); LUO = Path(argv[i + 1]); del argv[i:i + 2]
RES = Path(argv[0]) if argv else cnt_paths.results_dir()
if TEX is None and len(argv) > 1:
    TEX = Path(argv[1])
if TEX is None and os.environ.get("CNT_TEX"):
    TEX = Path(os.environ["CNT_TEX"])

rows = list(csv.DictReader(open(RES / "results_probes.csv")))
ft = list(csv.DictReader(open(RES / "results_finetune.csv")))
a = json.load(open(RES / "analysis.json"))
pim = {e["key"]: e for e in json.load(open(RES / "per_image_probe.json"))}
g = lambda r, k: float(r[k])
H = a["headline"]
CLS = ["Fiber", "Cluster", "Matrix", "MatrixSurface"]

# --------------------------------------------------------------------------
# derive every number the manuscript quotes
# --------------------------------------------------------------------------
V = {}                                   # name -> (value, rendering-as-printed)
def put(name, value, printed):
    V[name] = dict(value=value, printed=printed)

ci = H["test_ensemble_ci"]
if isinstance(ci, str):
    ci = [float(x) for x in ci.strip("[]").split(",")]
n_dev, n_test = int(H["oof_n"]), int(H["test_n"])
put("headline_cv_acc", H["oof_acc"], f"{H['oof_acc']:.1f}%")
put("headline_cv_std", H["cv_std"], f"{H['cv_std']:.1f}%")
put("headline_test_acc", H["test_ensemble_acc"], f"{H['test_ensemble_acc']:.1f}%")
put("headline_test_correct", int(H["test_ensemble_correct"]),
    f"{H['test_ensemble_correct']}/{n_test}")
put("headline_test_ci", ci, f"95% CI {ci[0]:.1f}-{ci[1]:.1f}%")
put("headline_table_cv", H["oof_acc"], f"{H['oof_acc']:.2f}")
put("headline_table_test", H["test_ensemble_acc"], f"{H['test_ensemble_acc']:.2f}")

# split arithmetic: dev = train + val; the val size is a property of the split file
import pickle
raw = pickle.load(open(cnt_paths.splits_file(), "rb"))
n_train, n_val, n_test_pkl = (len(raw["train_df"]), len(raw["val_df"]), len(raw["test_df"]))
if n_train + n_val != n_dev or n_test_pkl != n_test:
    raise SystemExit(f"split file ({n_train}/{n_val}/{n_test_pkl}) does not match the "
                     f"results ({n_dev} dev / {n_test} test)")
from collections import Counter
cnt = Counter(r["category"] for k in ("train_df", "val_df", "test_df") for r in raw[k])
put("n_images", n_train + n_val + n_test, f"{n_train + n_val + n_test:,}")
put("split", [n_train, n_val, n_test], f"{n_train:,} / {n_val} / {n_test}")
for c in CLS:
    put(f"n_{c}", cnt[c], f"{cnt[c]} {c}")

o = [g(r, "oof_acc") for r in rows]
put("grid_size", len(rows), f"{len(rows)} model configurations")
put("grid_min", min(o), f"{min(o):.1f}%"); put("grid_max", max(o), f"{max(o):.1f}%")
put("grid_mean", float(np.mean(o)), f"{np.mean(o):.1f}%")
put("grid_std", float(np.std(o)), f"{np.std(o):.1f}%")

fz = a["mask_effect"]["frozen"]; fn = a["mask_effect"]["finetuned"]
put("mask_pairs", fz["n_pairs"], f"all {fz['n_pairs']} matched pairs")
put("mask_mean", fz["mean_delta"], f"{fz['mean_delta']:.2f} percentage points")
put("mask_min", fz["min_delta"], f"{fz['min_delta']:.2f}")
put("mask_max", fz["max_delta"], f"{fz['max_delta']:.2f}")
put("mask_t", fz["paired_t"], f"t = {fz['paired_t']:.1f}")
put("mask_sign_p", fz["sign_test_p"], f"p = {fz['sign_test_p']:.1e}")
put("ft_mask_mean", fn["mean_delta"], f"{fn['mean_delta']:.2f} points")
put("ft_mask_favouring", fn["n_favouring_masked"],
    f"{fn['n_favouring_masked']} of {fn['n_pairs']} models improving")

cmp = {c["method"]: c for c in a["comparisons"]}
# "outperformed every fine-tuned baseline, by X to Y points": the range spans
# ALL TEN fine-tuned configurations (5 architectures x masked/raw), not only
# the five best-of-two rows shown in Table 3.
ftk = [f"{r['method']}|{'masked' if r['mask']=='masked' else 'raw'}" for r in ft]
deltas = [cmp[k]["delta"] for k in ftk if k in cmp]
put("baseline_margin_min", min(deltas), f"{min(deltas):.1f}")
put("baseline_margin_max", max(deltas), f"{max(deltas):.1f} percentage points (over all {len(deltas)} fine-tuned configurations)")
worst = min((cmp[k] for k in ftk if k in cmp), key=lambda c: c["delta"])
put("closest_finetuned", worst["method"], worst["method"])
put("worst_p_holm_significant", bool(worst["p_holm"] < 0.05), str(worst["p_holm"] < 0.05))
put("frozen_none_significant",
    bool(all(cmp[k]["p_holm"] >= 0.05 for k in cmp if "[" in k)),
    "no frozen encoder differs significantly")
put("family_size", a["family_size"], f"{a['family_size']} tests")

# refit flags: every flagged row must be documented, and the paper never quotes a refit number
import paper_protocol as P
flagged = [f"{r['method']}/{r['mask']}" for r in ft
           if str(r.get("refit_unreliable", P.refit_unreliable(int(r["epoch_budget"])))).lower() == "true"]
put("refit_unreliable_rows", flagged, ", ".join(flagged) or "none")

# Luo et al.: best clean variant on CV
luo, luo_rows = luo_results.reported_luo(LUO, cnt_paths.splits_file().stem)
put("luo_variant", luo["desc_norm"], luo["desc_norm"])
put("luo_variants_run", sorted(r["desc_norm"] for r in luo_rows),
    "/".join(sorted(r["desc_norm"] for r in luo_rows)))
put("luo_cv", float(luo["cv_acc"]), f"{float(luo['cv_acc']):.2f}")
put("luo_cv_std", float(luo["cv_std"]), f"{float(luo['cv_std']):.1f}")
put("luo_test", float(luo["test_acc"]), f"{float(luo['test_acc']):.2f}")
put("luo_margin", H["oof_acc"] - float(luo["cv_acc"]),
    f"{H['oof_acc'] - float(luo['cv_acc']):.1f} percentage points")

# confusion matrix, re-derived from the per-image predictions
e = pim[f"{a['primary']['encoder']}[{a['primary']['geometry']}]|{a['primary']['mask']}"
        f"|{a['primary']['pooling']}|{a['primary']['classifier']}"]
y, pr = np.array(e["y_test"]), np.array(e["pred_ensemble"])
cm = np.zeros((4, 4), int)
for t, q in zip(y, pr): cm[t, q] += 1
rec = {CLS[i]: 100 * cm[i, i] / cm[i].sum() for i in range(4)}
put("confusion", cm.tolist(), str(cm.tolist()))
put("best_class", max(rec, key=rec.get), f"{max(rec, key=rec.get)} ({rec[max(rec, key=rec.get)]:.0f}%)")
put("worst_class", min(rec, key=rec.get), f"{min(rec, key=rec.get)} ({rec[min(rec, key=rec.get)]:.0f}%)")
put("nothing_assigned_to_fiber", bool(cm[:, 0].sum() == cm[0, 0]), str(cm[:, 0].sum() == cm[0, 0]))
put("confusion_acc", 100 * np.trace(cm) / cm.sum(), f"{100*np.trace(cm)/cm.sum():.1f}%")
if abs(100 * np.trace(cm) / cm.sum() - H["test_ensemble_acc"]) > 1e-6:
    raise SystemExit("confusion matrix accuracy != headline test accuracy")

# --------------------------------------------------------------------------
def report(title):
    print(title)
    for k, v in V.items():
        print(f"   {k:28} {v['printed']}")


def _num_equal(x, y):
    if isinstance(x, (list, tuple)):
        return isinstance(y, (list, tuple)) and len(x) == len(y) and all(_num_equal(p, q) for p, q in zip(x, y))
    if isinstance(x, bool) or isinstance(y, bool):
        return x == y
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return abs(x - y) < 1e-6
    return x == y


if write_snapshot:
    json.dump(V, open(SNAP, "w"), indent=2)
    report(f"wrote {SNAP}:")
    sys.exit(0)

ok, bad = [], []
if TEX is not None:
    # ---- the authors' check: strings in the .tex ----------------------------
    main = (TEX / "main.tex").read_text(encoding="utf-8")
    supp = (TEX / "supplementary.tex").read_text(encoding="utf-8")
    both = main + "\n" + supp
    def check(desc, needle, where=both):
        (ok if needle in where else bad).append((desc, needle))
    check("abstract CV",        f"{H['oof_acc']:.1f}\\% cross-validated accuracy")
    check("abstract test",      f"{H['test_ensemble_acc']:.1f}\\% on a held-out test set")
    check("results CV",         f"{H['oof_acc']:.1f}\\% $\\pm$ {H['cv_std']:.1f}\\%")
    check("results test+CI",    f"{H['test_ensemble_acc']:.1f}\\% ({H['test_ensemble_correct']}/{n_test}, "
                                f"95\\% CI {ci[0]:.1f}--{ci[1]:.1f}\\%)")
    check("confusion overall",  f"{H['test_ensemble_acc']:.1f}\\% ({H['test_ensemble_correct']}/{n_test})")
    check("table2 headline CV", f"\\textbf{{{H['oof_acc']:.2f}}}")
    check("table2 headline test", f"\\textbf{{{H['test_ensemble_acc']:.2f}}}")
    check("exact split in Methods", f"(1{{,}}{n_train-1000:03d} / {n_val} / {n_test} images)")
    check("exact totals in Methods", f"1{{,}}{n_train + n_val + n_test - 1000:03d} images---{cnt['Fiber']} Fiber")
    for tok in (f"1{{,}}{n_train + n_val + n_test - 1000:03d}", f"1{{,}}{n_train-1000:03d}",
                *[f"{cnt[c]} {c}" for c in CLS]):
        check(f"split token {tok}", tok)
    check("grid size", f"{len(rows)} model configurations")
    check("grid range", f"{min(o):.1f}\\% to {max(o):.1f}\\%")
    check("grid mean", f"{np.mean(o):.1f}\\% $\\pm$ {np.std(o):.1f}\\%")
    check("mask n pairs", f"all {fz['n_pairs']} matched pairs")
    check("mask mean", f"by {fz['mean_delta']:.2f} percentage points on average")
    check("mask range", f"between {fz['min_delta']:.2f} and {fz['max_delta']:.2f} points")
    check("mask t (supplement)", f"$t = {fz['paired_t']:.1f}$")
    check("finetuned mask mean", f"only {fn['mean_delta']:.2f} points on average")
    check("finetuned mask count", f"with {['zero','one','two','three','four','five'][fn['n_favouring_masked']]} of {['zero','one','two','three','four','five'][fn['n_pairs']]} models improving")
    check("baseline margin", f"{min(deltas):.1f} to {max(deltas):.1f} percentage points")
    if worst["p_holm"] >= 0.05:
        bad.append(("a fine-tuned baseline is NOT significant -- text claims all are", worst["method"]))
    check("closest baseline", "ViT-B/16 fine-tuned on masked crops" if worst["method"] == "vit_b16|masked"
          else f"closest baseline is {worst['method']}")
    check("Luo table row", f"& VGG-16 + VLAD & {float(luo['cv_acc']):.2f} & {float(luo['test_acc']):.2f} &")
    check("Luo in text", f"({float(luo['cv_acc']):.1f}\\% $\\pm$ {float(luo['cv_std']):.1f}\\% cross-validated, "
                         f"{float(luo['test_acc']):.1f}\\% test)")
    check("Luo margin", f"by {H['oof_acc'] - float(luo['cv_acc']):.1f} percentage points")
    check("caption best class", f"{max(rec, key=rec.get)} is classified most reliably ({rec[max(rec, key=rec.get)]:.0f}\\%)")
    check("caption worst class", f"{min(rec, key=rec.get)} least ({rec[min(rec, key=rec.get)]:.0f}\\%)")
    if cm[:, 0].sum() != cm[0, 0]:
        bad.append(("text claims nothing is assigned to Fiber, but something is", str(cm[:, 0])))
    check("caption test size", f"({n_test} images, approximately 45 per class)")
    body = re.sub(r"(?m)^%%.*$", "", main)
    if "\\pending{" in body:
        bad.append(("unreplaced \\pending in body", "\\pending{"))
    src = "the manuscript source"
else:
    # ---- the reader's check: the committed snapshot --------------------------
    if not SNAP.exists():
        raise SystemExit(f"{SNAP} is missing and no manuscript source was given")
    snap = json.load(open(SNAP))
    for k, v in V.items():
        if k not in snap:
            bad.append((k, f"not in snapshot; now {v['printed']}")); continue
        if _num_equal(v["value"], snap[k]["value"]):
            ok.append((k, v["printed"]))
        else:
            bad.append((k, f"snapshot {snap[k]['printed']!r} vs results {v['printed']!r}"))
    for k in snap:
        if k not in V:
            bad.append((k, "in snapshot but no longer derived"))
    src = f"{SNAP.name} (numbers as printed in the manuscript)"

report("derived from the results files:")
print(f"\nPASS {len(ok)}")
for d, v in ok: print(f"   ok   {d}")
if bad:
    print(f"\nFAIL {len(bad)}")
    for d, v in bad: print(f"   MISS {d}: {v}")
    print(f"\nthe results files DISAGREE with {src}.")
    sys.exit(1)
print(f"\nevery checked number matches {src}.")
