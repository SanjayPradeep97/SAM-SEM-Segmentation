"""Read the Luo et al. baseline results (baseline/results/luo_results_*.json)
and pick the row the paper reports.

Selection rule: among the CLEAN-protocol runs for the repository's split file,
the descriptor-normalisation variant with the best CROSS-VALIDATED accuracy.
The held-out test accuracy is never consulted.  The JSON files are read rather
than luo_baseline_summary.csv so no value passes through a second rounding.
"""
import glob, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DIR = HERE.parent / "baseline" / "results"


def load_luo(results_dir=DEFAULT_DIR, split_stem="dataset_splits", earlystop="clean"):
    rows = []
    for f in sorted(glob.glob(str(Path(results_dir) / "luo_results_*.json"))):
        d = json.load(open(f))
        if Path(str(d["splits"])).stem != split_stem or d["earlystop"] != earlystop:
            continue
        rows.append(dict(file=Path(f).name, desc_norm=d["desc_norm"],
                         earlystop=d["earlystop"],
                         cv_acc=float(d["cv_mean_acc"]), cv_std=float(d["cv_std_acc"]),
                         cv_f1=float(d["cv_mean_f1"]), test_acc=float(d["test_acc"]),
                         test_correct=int(d["test_correct"]), test_n=int(d["test_n"]),
                         test_ci=(float(d["test_ci_low"]), float(d["test_ci_high"])),
                         test_f1=float(d["test_f1"]), rounds=int(d["final_rounds"])))
    if not rows:
        raise SystemExit(f"no Luo results for split {split_stem!r} / {earlystop} in {results_dir}")
    return rows


def reported_luo(results_dir=DEFAULT_DIR, split_stem="dataset_splits"):
    rows = load_luo(results_dir, split_stem, "clean")
    return max(rows, key=lambda r: r["cv_acc"]), rows


if __name__ == "__main__":
    best, rows = reported_luo()
    print(f"{'variant':9} {'CV acc':>8} {'sd':>5} {'test':>7} {'correct':>8}")
    for r in sorted(rows, key=lambda r: -r["cv_acc"]):
        print(f"{r['desc_norm']:9} {r['cv_acc']:8.2f} {r['cv_std']:5.2f} {r['test_acc']:7.2f} "
              f"{r['test_correct']:>4}/{r['test_n']}" + ("   <- reported (best CV)" if r is best else ""))
