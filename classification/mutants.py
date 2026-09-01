"""Re-run the audit's seven injected bugs against the hardened test suite.
v1 caught 1 of 7.  Anything that still passes is a hole."""
import subprocess, shutil, os, sys, tempfile, re

SRC = os.path.dirname(os.path.abspath(__file__))
MUT = {
 "M7 early-stop on the evaluation fold (the original bug, one level up)":
   ("paper_protocol.py",
    "(p_va, p_te), ep = fit(i_fit, i_es, [i_va, \"test\"], None, seed)",
    "(p_va, p_te), ep = fit(i_fit, i_va, [i_va, \"test\"], None, seed)"),
 "M1 scaler fit on all of dev":
   ("probe_fit.py", "sc = StandardScaler().fit(X_dev[train_idx])",
    "sc = StandardScaler().fit(X_dev)"),
 "M5 scaler also sees the test set":
   ("probe_fit.py", "sc = StandardScaler().fit(X_dev[train_idx])",
    "sc = StandardScaler().fit(np.vstack([X_dev[train_idx], X_test]))"),
 "M3 ensemble collapses to fold 1":
   ("paper_protocol.py", "ens = np.mean(test_probs_per_fold, axis=0)",
    "ens = test_probs_per_fold[0]"),
 "M8 best checkpoint never restored":
   ("probe_fit.py", "        if use_es and best_state is not None:",
    "        if False and best_state is not None:"),
 "M4 out-of-fold rows reversed":
   ("paper_protocol.py", "oof[i_va] = p_va", "oof[i_va] = p_va[::-1]"),
 "M2 budget = max instead of median":
   ("paper_protocol.py", "budget = int(round(float(np.median(best_epochs))))",
    "budget = int(max(best_epochs))"),
}

caught = 0
for name, (fn, old, new) in MUT.items():
    d = tempfile.mkdtemp()
    for f in ("paper_protocol.py", "probe_fit.py", "test_protocol.py",
              "cnt_paths.py", "encoder_bench.py", "gpu_boost.py", "cache_setup.py"):
        shutil.copy(os.path.join(SRC, f), d)
    # test_protocol locates the split file relative to the repo; the copy is
    # two levels down from nowhere, so point it at the real one explicitly.
    os.environ.setdefault("CNT_SPLITS", os.path.join(os.path.dirname(SRC), "splits",
                                                     "dataset_splits.pkl"))
    p = os.path.join(d, fn); s = open(p).read()
    if old not in s:
        print(f"  [SETUP-FAIL] {name}: anchor not found"); continue
    open(p, "w").write(s.replace(old, new, 1))
    r = subprocess.run([sys.executable, os.path.join(d, "test_protocol.py")],
                       capture_output=True, text=True, cwd=d, env=dict(os.environ))
    tail = [l for l in r.stdout.splitlines() if "passed," in l]
    fails = re.findall(r"\[FAIL\] (.+?)(?:  --|$)", r.stdout)
    ok = r.returncode != 0
    caught += ok
    print(f"  [{'CAUGHT' if ok else 'ESCAPED'}] {name}")
    print(f"             {tail[0].strip() if tail else 'crashed'}"
          + (f"  ->  {fails[0][:70]}" if fails else ""))
    shutil.rmtree(d)
print(f"\n  {caught}/{len(MUT)} mutants caught   (v1 of this suite caught 1/7)")
sys.exit(0 if caught == len(MUT) else 1)
