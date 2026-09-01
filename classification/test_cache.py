"""Round-trip tests for the download/decode caches."""
import sys, os, tempfile, shutil, time
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BASE = Path(tempfile.mkdtemp())
os.environ["CNT_BASE"] = str(BASE)
import cache_setup
cache_setup.setup(BASE, verbose=False)
import finetune_baselines as FB

PASS, FAIL = [], []
def check(n, c, d=""):
    (PASS if c else FAIL).append(n)
    print(f"  [{'PASS' if c else 'FAIL'}] {n}" + (f"  -- {d}" if d else ""))

# synthetic stand-ins for the 1350x1040 TIFFs
img_dir = BASE / "imgs"; img_dir.mkdir()
rng = np.random.default_rng(0)
rows = []
for i in range(12):
    a = rng.integers(0, 255, (1040, 1350), dtype=np.uint8)
    m = np.zeros((1040, 1350), np.uint8); m[300:700, 400:900] = 255
    ip, mp = img_dir / f"im{i}.png", img_dir / f"im{i}_mask.png"
    Image.fromarray(a).save(ip); Image.fromarray(m).save(mp)
    rows.append({"image": ip, "mask": mp, "y": i % 4})

print("\n1. SHRINK NEVER UPSCALES")
small = np.zeros((50, 80), np.uint8)
check("a 80px image is left alone at max_side=384",
      FB._shrink(small, 384).shape == (50, 80), str(FB._shrink(small, 384).shape))
big = np.zeros((1040, 1350), np.uint8)
out = FB._shrink(big, 384)
check("a 1350px image is capped at 384 on the long side",
      max(out.shape) == 384, str(out.shape))
check("aspect ratio preserved",
      abs(out.shape[0] / out.shape[1] - 1040 / 1350) < 0.01, str(out.shape))
check("max_side=None is a no-op", FB._shrink(big, None).shape == (1040, 1350))

print("\n2. DISK CACHE ROUND TRIP")
FB._PREP_CACHE.clear()
t0 = time.time(); a1 = FB.prepare(rows, True, "unit"); t_cold = time.time() - t0
FB._PREP_CACHE.clear()                       # force the disk path, not memory
t0 = time.time(); a2 = FB.prepare(rows, True, "unit"); t_warm = time.time() - t0
check("same number of items", len(a1) == len(a2) == len(rows))
check("pixels identical after a cache round trip",
      all(np.array_equal(x[0], y[0]) for x, y in zip(a1, a2)))
check("labels identical", [y for _, y in a1] == [y for _, y in a2])
check("warm read is faster than a cold decode",
      t_warm < t_cold, f"cold {t_cold:.2f}s vs warm {t_warm:.2f}s")
cached = list((BASE / "_cache" / "prep").glob("*.npz"))
check("exactly one cache file written", len(cached) == 1, str([c.name for c in cached]))

print("\n3. CACHE KEY IS CONTENT-ADDRESSED")
p_mask = cache_setup.prep_cache_path("unit", True, rows, 384)
p_raw = cache_setup.prep_cache_path("unit", False, rows, 384)
p_side = cache_setup.prep_cache_path("unit", True, rows, 256)
p_short = cache_setup.prep_cache_path("unit", True, rows[:-1], 384)
p_order = cache_setup.prep_cache_path("unit", True, list(reversed(rows)), 384)
check("masked and raw differ", p_mask != p_raw)
check("a different max_side differs", p_mask != p_side)
check("a different row count differs", p_mask != p_short)
check("a different row ORDER differs  [folds index into this list]",
      p_mask != p_order)
check("identical inputs give an identical key",
      p_mask == cache_setup.prep_cache_path("unit", True, rows, 384))

print("\n4. RAW VARIANT CACHES SEPARATELY")
FB._PREP_CACHE.clear()
r1 = FB.prepare(rows, False, "unit")
check("raw images keep full resolution up to max_side",
      max(r1[0][0].shape) == 384, str(r1[0][0].shape))
check("two cache files now exist",
      len(list((BASE / "_cache" / "prep").glob("*.npz"))) == 2)

print("\n5. STALE CACHE IS NOT REUSED")
FB._PREP_CACHE.clear()
bad = BASE / "_cache" / "prep" / p_mask.name
np.savez(bad, y=np.array([0, 1], dtype=np.int16),
         i0=np.zeros((4, 4), np.uint8), i1=np.zeros((4, 4), np.uint8))
got = FB.prepare(rows, True, "unit")
check("a cache with the wrong row count is rebuilt, not used",
      len(got) == len(rows), f"got {len(got)}")

shutil.rmtree(BASE, ignore_errors=True)
print(f"\n  {len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
