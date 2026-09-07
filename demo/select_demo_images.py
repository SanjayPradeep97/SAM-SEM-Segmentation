"""Choose the demo image set from the held-out test predictions.

The demo is meant to show the combined segment-then-classify pipeline on
cases that are not ambiguous, so it deliberately picks EASY images: the 20
test images per class (Fiber, Cluster, Matrix) that the largest number of the
118 evaluated configurations classified correctly.  Only held-out test images
are eligible, because the shipped classifier head was trained on the
development images and must not have seen anything the demo shows it.

This script exists so the selection is reproducible and auditable; its output
`demo_images.csv` is committed.  Re-running it against the shipped results
files regenerates the same list.

usage:  python select_demo_images.py [<results dir>] [<splits pkl>]
"""
import csv, json, pickle, sys
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "results"
SPLITS = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "splits" / "dataset_splits.pkl"
CLASSES = ["Fiber", "Cluster", "Matrix"]          # MatrixSurface is left out on purpose
PER_CLASS = 20

raw = pickle.load(open(SPLITS, "rb"))
test_rows = raw["test_df"]                         # same order as every results file
entries = (json.load(open(RES / "per_image_probe.json"))
           + json.load(open(RES / "per_image_finetune.json")))
y = np.array(entries[0]["y_test"])
if not all(int(r["category_id"]) == int(t) for r, t in zip(test_rows, y)):
    raise SystemExit("test labels in the results files do not match the split file")

n_right = np.zeros(len(y), int)
for e in entries:
    if list(e["y_test"]) != list(y):
        raise SystemExit(f"{e['key']}: y_test differs")
    n_right += (np.array(e["pred_ensemble"]) == y)

primary = next(e for e in entries
               if e["key"] == "dinov2_b14[g37_L1-3-6-9-11]|masked|avg+max|mlp")
prim_ok = np.array(primary["pred_ensemble"]) == y

# only images that are in the public Dataverse record are eligible, so every
# demo image can be traced to its deposited original (splits/dataverse_files.txt)
DATAVERSE = {l.strip() for l in open(HERE.parent / "splits" / "dataverse_files.txt") if not l.startswith("#")}

picked = []
for cls in CLASSES:
    cand = [(int(n_right[i]), r["filename"], i) for i, r in enumerate(test_rows)
            if r["category"] == cls and prim_ok[i] and r["filename"] in DATAVERSE]
    cand.sort(key=lambda t: (-t[0], t[1]))         # most configs right, then by name
    for n, fn, i in cand[:PER_CLASS]:
        picked.append(dict(filename=fn, category=cls, split="test",
                           configs_correct=n, configs_total=len(entries)))

out = HERE / "demo_images.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(picked[0]))
    w.writeheader(); w.writerows(picked)
print(f"wrote {out}: {Counter(p['category'] for p in picked)}")
print(f"configs correct per image: min {min(p['configs_correct'] for p in picked)}"
      f" / {len(entries)}")
