"""Emit the LaTeX bodies of Tables 2 and 3 straight from the results files.

Every number in the manuscript's two results tables is produced here, so the
tables cannot drift from results_probes.csv / results_finetune.csv /
analysis.json / the Luo baseline summary.  Re-run after any change to the
pipeline and paste the output.

Selection rules, stated once:
  * the headline row is the PRE-SPECIFIED configuration (paper_protocol.PRIMARY),
    never the best-scoring one;
  * every other frozen encoder shows its best masked configuration BY
    CROSS-VALIDATED ACCURACY;
  * every fine-tuned model shows the better of masked/raw BY CROSS-VALIDATED
    ACCURACY;
  * the Luo et al. row is the descriptor-normalisation variant with the best
    CROSS-VALIDATED accuracy under the clean protocol.
The held-out test column is never consulted for any of these choices.

usage: python make_tables.py [<results dir>] [--luo <dir of luo_results_*.json>]
"""
import csv, json, sys, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cnt_paths
import luo_results

argv = sys.argv[1:]
LUO = luo_results.DEFAULT_DIR
if "--luo" in argv:
    i = argv.index("--luo"); LUO = Path(argv[i + 1]); del argv[i:i + 2]
RES = Path(argv[0]) if argv else cnt_paths.results_dir()

rows = list(csv.DictReader(open(RES / "results_probes.csv")))
ft   = list(csv.DictReader(open(RES / "results_finetune.csv")))
a    = json.load(open(RES / "analysis.json"))
cmp  = {c["method"]: c for c in a["comparisons"]}
g    = lambda r, k: float(r[k])
PRIM = a["primary"]
GEOM = PRIM["geometry"]
SPLIT_STEM = cnt_paths.splits_file().stem


def luo_row(path=LUO, split_stem=SPLIT_STEM):
    """Best clean-protocol Luo variant for this split, chosen on CV accuracy."""
    best, _ = luo_results.reported_luo(path, split_stem)
    return best


def find(meth, geom, mask, pool, clf):
    for r in rows:
        if (r["method"], r["geometry"], r["mask"], r["pooling"], r["classifier"]) \
           == (meth, geom, mask, pool, clf):
            return r
    raise KeyError((meth, geom, mask, pool, clf))


# ---------------------------------------------------------------- Table 2
POOLTeX = {"avg": "Avg", "max": "Max", "avg+max": "A+M"}
PLAN = [("DINOv2", "dinov2_b14", GEOM,
         [("masked","avg+max","mlp",True), ("masked","avg","mlp",False),
          ("masked","avg","linear",False), ("nomask","avg","mlp",False)]),
        ("SAM", "sam_b16", "g32_t5",
         [("masked","avg+max","mlp",False), ("masked","avg","mlp",False),
          ("masked","avg","linear",False), ("nomask","avg","mlp",False)])]
t2 = []
for label, meth, geom, specs in PLAN:
    for mask, pool, clf, bold in specs:
        r = find(meth, geom, mask, pool, clf)
        cv, te, f1 = g(r,"oof_acc"), g(r,"test_ensemble_acc"), g(r,"test_ensemble_f1")
        b = (lambda s: rf"\textbf{{{s}}}") if bold else (lambda s: s)
        t2.append(f"{label:6} & {'Yes' if mask=='masked' else 'No ':3} & {POOLTeX[pool]:3} & "
                  f"{clf.upper() if clf=='mlp' else 'Linear':6} & {b(f'{cv:.2f}')} & "
                  f"{b(f'{te:.2f}')} & {b(f'{f1:.3f}')} \\\\")
    if label == "DINOv2":
        t2.append(r"\midrule")

# ---------------------------------------------------------------- Table 3
NAME = {"dinov2_b14":"DINOv2 ViT-B/14 + mask-guided (this work)",
        "dinov2_reg":"DINOv2 ViT-B/14 + registers", "dinov3_b16":"DINOv3 ViT-B/16",
        "sam_b16":"SAM ViT-B/16", "sam2_hiera":r"SAM\,2 Hiera",
        "resnet50":"ResNet-50 (frozen)", "supvit_b16":"Supervised ViT-B/16 (frozen)",
        "convnextv2_b":"ConvNeXt-V2 (frozen)"}
FTNAME = {"resnet50":"ResNet-50","vit_b16":"ViT-B/16","convnextv2":"ConvNeXt-V2",
          "yolo11":"YOLOv11-cls","yolo26":"YOLO26-cls"}


def ptex(key):
    c = cmp.get(key)
    if c is None: return "---"
    p = c["p_holm"]
    return r"$<$0.001" if p < 0.001 else f"{p:.3f}"


frozen = []
for meth in NAME:
    if meth == PRIM["encoder"]:                      # headline: the PRE-SPECIFIED config only
        r = find(meth, GEOM, PRIM["mask"], PRIM["pooling"], PRIM["classifier"])
    else:                                            # others: best masked config by CV
        cand = [x for x in rows if x["method"] == meth and x["mask"] == "masked"]
        r = max(cand, key=lambda x: g(x, "oof_acc"))
    key = f"{r['method']}[{r['geometry']}]|{r['mask']}|{r['pooling']}|{r['classifier']}"
    frozen.append((NAME[meth], g(r,"oof_acc"), g(r,"test_ensemble_acc"),
                   "---" if meth == PRIM["encoder"] else ptex(key),
                   meth == PRIM["encoder"]))
frozen.sort(key=lambda t: -t[1])

fine = []
for meth in FTNAME:
    cand = [x for x in ft if x["method"] == meth]
    r = max(cand, key=lambda x: g(x, "oof_acc"))     # masked vs raw: chosen on CV
    tag = "masked" if r["mask"] == "masked" else "raw"
    key = f"{meth}|{tag}"
    fine.append((f"{FTNAME[meth]} ({tag}, fine-tuned)", g(r,"oof_acc"),
                 g(r,"test_ensemble_acc"), ptex(key), False))
fine.sort(key=lambda t: -t[1])

luo = luo_row()
t3 = []
for nm, cv, te, p, bold in frozen:
    b = (lambda s: rf"\textbf{{{s}}}") if bold else (lambda s: s)
    t3.append(f"{nm} & Frozen & {b(f'{cv:.2f}')} & {b(f'{te:.2f}')} & {p} \\\\")
t3.append(r"\midrule")
for nm, cv, te, p, _ in fine:
    t3.append(f"{nm} & Fine-tuned & {cv:.2f} & {te:.2f} & {p} \\\\")
t3.append(r"\midrule")
t3.append(rf"Luo et al.\cite{{Luo_Wang}} (re-implementation) & VGG-16 + VLAD & "
          rf"{float(luo['cv_acc']):.2f} & {float(luo['test_acc']):.2f} & --- \\")

if __name__ == "__main__":
    print("%%% ---------------- TABLE 2 body ----------------")
    print("\n".join(t2))
    print("\n%%% ---------------- TABLE 3 body ----------------")
    print("\n".join(t3))
    print(f"\n%%% Luo row: desc_norm={luo['desc_norm']} (best clean variant on CV, "
          f"from {luo['file']})")
