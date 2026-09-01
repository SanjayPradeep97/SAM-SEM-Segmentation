"""Assert that encoder_bench and paper_results agree, byte for byte, on every
feature-cache filename -- for the default configuration AND for every
non-default variant that changes the features.

They build the name independently.  If they ever disagree, stage 3 either
aborts hours after stage 1 finished, or -- far worse -- silently loads the
features of the *other* configuration.  This runs in seconds; run it before
every long extraction.

Variants checked, beyond the split file and geometry:
  * pyramid input size          (--pyramid-img 384 vs the default 512)
  * random weights              (ENCBENCH_NO_PRETRAINED=1, smoke tests only)
  * a checkpoint override       (USAM_CHECKPOINT, only meaningful for usam_b)
Each must produce a DIFFERENT name from the default, and the two scripts must
agree on all of them.

usage: python check_cache_agreement.py [--splits <pkl>] [--no-data]
       --no-data skips the split-loading step (no CNT_BASE needed)
"""
import argparse, os, sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cnt_paths

ap = argparse.ArgumentParser()
ap.add_argument("--splits", default=str(cnt_paths.splits_file()))
ap.add_argument("--no-data", action="store_true")
a = ap.parse_args()

import encoder_bench as EB
import paper_results as PR

PR.SPLIT_STEM = Path(a.splits).stem
if PR.BENCH is None:                       # no CNT_BASE: names only, any root will do
    PR.BENCH = Path("bench")


def cfg_for(grid, taps, pyramid_img=512):
    args = SimpleNamespace(
        target_grid=grid, pyramid_img=pyramid_img, n_taps=5, taps=taps,
        batch_size=16, epochs=100, earlystop="clean", act_budget=200e6,
        throttle_ms=0, splits=Path(a.splits))
    return EB.Cfg(args, PR.BENCH)


bad, n = 0, 0
print(f"split stem: {PR.SPLIT_STEM}\n")
hdr = f"{'variant':12} {'encoder':14} {'geometry':18} {'encoder_bench':58} {'paper_results':58} ok"
print(hdr); print("-" * len(hdr))


def compare(variant, enc, geom, cfg):
    global bad, n
    n1 = cfg.cache(enc).name
    n2 = PR.cache_path(enc, geom).name
    ok = n1 == n2
    bad += (not ok); n += 1
    print(f"{variant:12} {enc:14} {geom[0]:18} {n1:58} {n2:58} {'OK' if ok else 'MISMATCH'}")
    return n1


def sweep(variant, pyramid_img=512):
    names = {}
    PR.PYRAMID_IMG = pyramid_img
    for enc in PR.ENCODERS + PR.OPTIONAL_ENCODERS:
        for geom in PR.geometries_for(enc):
            taps = ",".join(map(str, geom[2])) if isinstance(geom[2], list) else None
            names[(enc, geom[0])] = compare(variant, enc, geom,
                                            cfg_for(geom[1], taps, pyramid_img))
    PR.PYRAMID_IMG = 512
    return names


default = sweep("default")

# --- pyramid input size ------------------------------------------------------
pyr = sweep("pyramid384", pyramid_img=384)
for k in default:
    if pyr[k] == default[k]:
        print(f"FAIL: pyramid_img not in cache key for {k}"); bad += 1

# --- random weights ------------------------------------------------------------
os.environ["ENCBENCH_NO_PRETRAINED"] = "1"
rnd = sweep("random")
del os.environ["ENCBENCH_NO_PRETRAINED"]
for k in default:
    if rnd[k] == default[k]:
        print(f"FAIL: random-weights flag not in cache key for {k}"); bad += 1

# --- checkpoint override -------------------------------------------------------
os.environ["USAM_CHECKPOINT"] = "some_micro_sam_weights.pt"
ck = sweep("ckpt")
del os.environ["USAM_CHECKPOINT"]
if ck[("usam_b", "g32_t5")] == default[("usam_b", "g32_t5")]:
    print("FAIL: checkpoint override not in cache key for usam_b"); bad += 1
for k in default:                       # and it must NOT touch the other encoders
    if k[0] != "usam_b" and ck[k] != default[k]:
        print(f"FAIL: checkpoint override leaked into {k}"); bad += 1

print()
if bad:
    print(f"FAIL: {bad} problem(s) across {n} name comparisons.")
    sys.exit(1)
print(f"all {n} cache names agree, and every variant changes the name. OK")

if not a.no_data:
    if cnt_paths.base(required=False) is None:
        print("CNT_BASE not set: skipping the split-loading check (pass --no-data to silence)")
        sys.exit(0)
    sp = EB.load_splits(Path(a.splits))
    cnt = {k: len(v) for k, v in sp.items()}
    print(f"splits load clean: {cnt}  total {sum(cnt.values())}")
