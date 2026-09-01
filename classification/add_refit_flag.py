"""Add the `refit_unreliable` column to results CSVs written before the flag
existed, using the rule in paper_protocol.refit_unreliable().

The shipped results_finetune.csv / results_probes.csv were produced by a run
that predates the flag; this script added it, and paper_results.py now writes
it natively.  Idempotent: rows that already carry the column are re-derived
from their epoch budget, so the value can never disagree with the rule.

usage: python add_refit_flag.py [<results dir>]
"""
import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cnt_paths, paper_protocol as P

RES = Path(sys.argv[1]) if len(sys.argv) > 1 else cnt_paths.results_dir()
for name in ("results_probes.csv", "results_finetune.csv"):
    p = RES / name
    rows = list(csv.DictReader(open(p, newline="")))
    fields = list(rows[0].keys())
    if "refit_unreliable" not in fields:
        i = fields.index("test_refit_f1") + 1        # right after the refit numbers
        fields.insert(i, "refit_unreliable")
    n_flag = 0
    for r in rows:
        r["refit_unreliable"] = P.refit_unreliable(int(r["epoch_budget"]))
        n_flag += r["refit_unreliable"]
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"{name}: {n_flag}/{len(rows)} rows flagged refit_unreliable "
          f"(epoch_budget < {P.REFIT_MIN_BUDGET})")
    for r in rows:
        if r["refit_unreliable"]:
            print(f"    {r['method']:14} {r['geometry']:16} {r['mask']:7} {r['pooling']:8} "
                  f"{r['classifier']:10} budget {r['epoch_budget']:>2}  "
                  f"ensemble {float(r['test_ensemble_acc']):5.1f}%  refit {float(r['test_refit_acc']):5.1f}%")
