"""Report whether a data root has everything the benchmark needs.

    set CNT_BASE=<data root>
    python classification/check_data.py

Checks, for each of the 1,785 images in splits/dataset_splits.pkl:
  * the image file exists under <CNT_BASE>/NIOSH Dataset/CNT-<class>/
  * its mask exists (CNT_MASKS or <CNT_BASE>/NIOSH Dataset/Masks/masks/)
  * whether the image is in the public Dataverse record (splits/dataverse_files.txt)
and prints counts per class and split, then exits 1 if anything is missing.
With --md5 it also prints an MD5 over the sorted list of (filename, size) so
two data roots can be compared at a glance.
"""
import csv, hashlib, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cnt_paths

split = list(csv.DictReader(open(HERE.parent / "splits" / "dataset_splits.csv")))
dv = {l.strip() for l in open(HERE.parent / "splits" / "dataverse_files.txt") if not l.startswith("#")}
base = cnt_paths.base()
print(f"data root   {base}")
print(f"images      {cnt_paths.images_dir()}")
print(f"masks       {cnt_paths.masks_dir()}\n")

miss_img, miss_msk, not_dv = [], [], []
sizes = []
for r in split:
    ip = cnt_paths.image_path(r["category"], r["filename"])
    mp = cnt_paths.mask_path(r["filename"])
    if not ip.exists():
        miss_img.append(r)
    else:
        sizes.append((r["filename"], ip.stat().st_size))
    if not mp.exists():
        miss_msk.append(r)
    if r["filename"] not in dv:
        not_dv.append(r)

n = len(split)
print(f"{'':14} {'images':>8} {'masks':>8} {'on Dataverse':>13}")
for cls in ("Fiber", "Cluster", "Matrix", "MatrixSurface"):
    tot = sum(r["category"] == cls for r in split)
    mi = sum(r["category"] == cls for r in miss_img)
    mm = sum(r["category"] == cls for r in miss_msk)
    nd = sum(r["category"] == cls for r in not_dv)
    print(f"{cls:14} {tot-mi:4d}/{tot:<4d} {tot-mm:4d}/{tot:<4d} {tot-nd:6d}/{tot:<4d}")
print(f"{'total':14} {n-len(miss_img):4d}/{n:<4d} {n-len(miss_msk):4d}/{n:<4d} {n-len(not_dv):6d}/{n:<4d}")
print(f"\nsplit images absent from the Dataverse record: {len(not_dv)} "
      f"({dict(Counter(r['split'] for r in not_dv))}); see splits/not_on_dataverse.csv")

if "--md5" in sys.argv and sizes:
    h = hashlib.md5("\n".join(f"{f}\t{s}" for f, s in sorted(sizes)).encode()).hexdigest()
    print(f"md5 over (filename, size) of the {len(sizes)} images present: {h}")

if miss_img or miss_msk:
    print(f"\nMISSING: {len(miss_img)} images, {len(miss_msk)} masks")
    for r in (miss_img + miss_msk)[:10]:
        print(f"   {r['filename']}")
    sys.exit(1)
print("\nevery image and mask of the split is present.")
