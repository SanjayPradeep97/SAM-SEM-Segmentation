#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Driver for the Luo et al. baseline.

Runs the baseline for every descriptor-normalisation option and BOTH
early-stopping protocols on the repository's split file, keeps a single global
progress file with a percentage and an ETA at all times, and collects
everything into one CSV (results/luo_baseline_summary.csv).

  none / l2 / blockl2   the three descriptor normalisations luo_baseline.py
                        documents.  The paper reports the one with the best
                        CROSS-VALIDATED accuracy under the clean protocol.

  clean       inner split picks the boosting rounds, then refit on 100%.
              The scored set is never used for model selection.
  leaky       reproduces the protocol of the withdrawn original analysis,
              where the set being scored also picks the stopping point.
              Reported only to quantify the bias; never publish it.

USAGE
  set CNT_BASE=<data root>
  python run_all.py                     # none + l2 + blockl2, clean + leaky (~1.5 h on a GPU)
  python run_all.py --norms l2          # one normalisation
  python run_all.py --report            # rebuild the summary, run nothing

WATCHING IT
  the file  <out-dir>/progress.json  is rewritten about once a second with
  overall_percent, the current stage, and both a stage ETA and a job ETA.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "classification"))
import cnt_paths

SCRIPT = HERE / "luo_baseline.py"
SPLITS = cnt_paths.splits_file()
STEM = SPLITS.stem

# rough per-job seconds on an RTX 5080; used only to weight the progress bar,
# so the percentage advances evenly rather than lurching between stages
EST = {"features": 210.0, "classify": 110.0}


def fmt(sec):
    if sec is None or sec != sec:
        return "--"
    sec = int(sec)
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def write_progress(path, **kw):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            json.dump(kw, fh, indent=2)
        tmp.replace(path)
    except Exception:
        pass


def collect(out_dir):
    rows = []
    for f in sorted(out_dir.glob("luo_results_*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        rows.append({
            "splits": Path(d["splits"]).stem,
            "desc_norm": d["desc_norm"],
            "earlystop": d["earlystop"],
            "cv_acc": round(d["cv_mean_acc"], 2),
            "cv_std": round(d["cv_std_acc"], 2),
            "cv_f1": round(d["cv_mean_f1"], 4),
            "test_acc": round(d["test_acc"], 2),
            "test_correct": f"{d['test_correct']}/{d['test_n']}",
            "test_ci95": f"[{d['test_ci_low']:.1f},{d['test_ci_high']:.1f}]",
            "test_f1": round(d["test_f1"], 4),
            "rounds": d["final_rounds"],
        })
    return rows


def report(out_dir):
    rows = collect(out_dir)
    if not rows:
        print("no results yet.")
        return
    out = out_dir / "luo_baseline_summary.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 100)
    print("LUO ET AL. (2021) BASELINE - ALL RUNS")
    print("=" * 100)
    hdr = list(rows[0].keys())
    widths = [max(len(h), max(len(str(r[h])) for r in rows)) + 2 for h in hdr]
    print("".join(f"{h:>{w}}" for h, w in zip(hdr, widths)))
    print("-" * sum(widths))
    for r in sorted(rows, key=lambda r: (r["splits"], r["desc_norm"], r["earlystop"])):
        print("".join(f"{str(r[h]):>{w}}" for h, w in zip(hdr, widths)))

    print("\nHEADLINE (clean protocol only -- the leaky rows exist to size the bias):")
    for stem in sorted({r["splits"] for r in rows}):
        cand = [r for r in rows if r["splits"] == stem and r["earlystop"] == "clean"]
        if not cand:
            continue
        best = max(cand, key=lambda r: r["cv_acc"])      # selected on CV, never test
        print(f"  {stem:24s} desc_norm={best['desc_norm']:8s} "
              f"CV {best['cv_acc']:5.2f}% +/- {best['cv_std']:.2f}   "
              f"Test {best['test_acc']:5.2f}% {best['test_correct']} CI {best['test_ci95']}"
              f"   <- the reported row")
        l = [r for r in rows if r["splits"] == stem and r["earlystop"] == "leaky"
             and r["desc_norm"] == best["desc_norm"]]
        if l:
            d = l[0]["test_acc"] - best["test_acc"]
            print(f"  {stem:24s} early-stopping leak on test: "
                  f"{d:+.2f} pp ({l[0]['test_acc']:.2f} leaky vs {best['test_acc']:.2f} clean)")
    print(f"\n  -> {out}")


def preflight():
    """Check the interpreter has everything BEFORE launching six jobs that will
    each fail identically. A missing package should cost one clear line, not
    six stack traces."""
    need = ["torch", "torchvision", "xgboost", "sklearn", "numpy", "PIL"]
    missing = []
    for m in need:
        try:
            __import__(m)
        except Exception:
            missing.append(m)
    if missing:
        print("=" * 78)
        print("PREFLIGHT FAILED - nothing was run")
        print("=" * 78)
        print(f"  interpreter : {sys.executable}")
        print(f"  missing     : {', '.join(missing)}")
        print()
        if "torch" in missing:
            print("  This interpreter has no PyTorch. If you trained the DINOv2 models on")
            print("  this machine, the environment that has it is probably a named conda")
            print("  env rather than base:")
            print()
            print("      conda env list")
            print("      conda activate <env-with-torch>")
            print('      python -c "import torch;print(torch.cuda.is_available())"')
        else:
            print(f"      pip install {' '.join(m for m in missing if m != 'torch')}")
        print()
        return False

    import torch
    if torch.cuda.is_available():
        print(f"  device      : cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("  device      : CPU ONLY -- torch.cuda.is_available() is False.")
        print("                The run is correct but ~3x slower (~40 min, not ~15).")
        print("                On an RTX 5080 (Blackwell, sm_120) this usually means")
        print("                the torch build predates CUDA 12.8.")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="VLAD caches (default <CNT_BASE>/Luo Baseline)")
    ap.add_argument("--results-dir", type=Path, default=HERE / "results")
    ap.add_argument("--norms", nargs="+", default=["none", "l2", "blockl2"],
                    choices=["none", "l2", "blockl2"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-init", type=int, default=10)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    a.results_dir.mkdir(parents=True, exist_ok=True)
    if a.report:
        report(a.results_dir)
        return
    if a.out_dir is None:
        a.out_dir = cnt_paths.base() / "Luo Baseline"
    a.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("PREFLIGHT")
    print("=" * 78)
    if not preflight():
        sys.exit(1)

    # ---- build the plan
    jobs = []
    for sname, spath in [(STEM, SPLITS)]:
        if not spath.exists():
            raise SystemExit(f"split file missing: {spath}")
        for norm in a.norms:
            common = ["--splits", str(spath), "--out-dir", str(a.out_dir),
                      "--results-dir", str(a.results_dir),
                      "--desc-norm", norm, "--device", a.device,
                      "--n-init", str(a.n_init)]
            jobs.append((f"{sname}/{norm}/features", common + ["--stage", "features"],
                         EST["features"]))
            for es in ("clean", "leaky"):
                jobs.append((f"{sname}/{norm}/{es}",
                             common + ["--stage", "classify", "--earlystop", es],
                             EST["classify"]))
    if not jobs:
        print("nothing to run.")
        return

    total = sum(j[2] for j in jobs)
    prog_path = a.out_dir / "progress.json"
    plan = [{"label": j[0], "est_s": j[2]} for j in jobs]
    with open(a.out_dir / "plan.json", "w") as fh:
        json.dump({"jobs": plan, "total_est_s": total}, fh, indent=2)

    print("=" * 78)
    print("PLAN")
    print("=" * 78)
    for j in jobs:
        print(f"  {j[0]:34s} est {fmt(j[2])}")
    print(f"  {'TOTAL':34s} est {fmt(total)}")
    print(f"  progress file: {prog_path}")
    print("=" * 78)

    t_start = time.time()
    offset = 0.0
    failed = []
    for i, (label, args_i, est) in enumerate(jobs):
        w = est / total
        elapsed = time.time() - t_start
        write_progress(prog_path, label=label, stage="starting",
                       overall_percent=round(100 * offset, 2),
                       job_index=i + 1, jobs_total=len(jobs),
                       elapsed_s=round(elapsed, 1),
                       overall_eta_s=round(elapsed / offset - elapsed, 1) if offset > 1e-6
                       else round(total - elapsed, 1),
                       updated=time.strftime("%Y-%m-%d %H:%M:%S"))
        print("\n" + "#" * 78)
        print(f"# JOB {i+1}/{len(jobs)}  {label}"
              f"   (overall {100*offset:.1f}% done, elapsed {fmt(elapsed)})")
        print("#" * 78, flush=True)
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, str(SCRIPT)] + args_i +
            ["--progress-file", str(prog_path), "--job-offset", f"{offset}",
             "--job-weight", f"{w}", "--job-label", label])
        if r.returncode != 0:
            print(f"!! FAILED: {label}")
            failed.append(label)
            if time.time() - t0 < 15:
                print("\n   Job died immediately -- this is a setup error, not a data\n"
                      "   problem, and the remaining jobs would fail the same way.\n"
                      "   Aborting the plan. Fix the error above and re-run.")
                break
        print(f"   [{fmt(time.time() - t0)}]")
        offset += w

    el = time.time() - t_start
    write_progress(prog_path, label="complete", stage="done", overall_percent=100.0,
                   job_index=len(jobs), jobs_total=len(jobs),
                   elapsed_s=round(el, 1), overall_eta_s=0,
                   failed=failed, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
    report(a.results_dir)
    print(f"\n  wall time: {fmt(el)}" + (f"   FAILED: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
