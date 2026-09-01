"""Put the 60 demo micrographs into demo/images/<class>/.

The images are NOT in this repository: they belong to the NIOSH TEM dataset
on Harvard Dataverse (doi:10.7910/DVN/5O0SF7, CC BY-NC 4.0).  This script
fetches exactly the files listed in demo_images.csv, either

  * from a local copy of the dataset (set CNT_BASE, see classification/cnt_paths.py)
        python fetch_demo_images.py
  * or straight from Dataverse, one file at a time (about 85 MB in total)
        python fetch_demo_images.py --download

Either way the result is demo/images/Fiber/*.tif etc., which demo_app.py reads.
Nothing else in the dataset is touched.
"""
import argparse, csv, json, shutil, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "classification"))
import cnt_paths

MANIFEST = HERE / "demo_images.csv"
OUT = HERE / "images"
DOI = "doi:10.7910/DVN/5O0SF7"
API = "https://dataverse.harvard.edu/api"

ap = argparse.ArgumentParser()
ap.add_argument("--download", action="store_true",
                help="fetch from Harvard Dataverse instead of a local copy")
a = ap.parse_args()

rows = list(csv.DictReader(open(MANIFEST)))
todo = [r for r in rows if not (OUT / r["category"] / r["filename"]).exists()]
print(f"{len(rows)} images in the manifest, {len(rows) - len(todo)} already present")
if not todo:
    sys.exit(0)

if not a.download:
    if cnt_paths.base(required=False) is None:
        raise SystemExit("CNT_BASE is not set.  Either point it at a local copy of the\n"
                         "dataset, or run with --download to fetch the 60 files from Dataverse.")
    for r in todo:
        src = cnt_paths.image_path(r["category"], r["filename"])
        if not src.exists():
            raise SystemExit(f"missing: {src}")
        dst = OUT / r["category"] / r["filename"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print(f"copied {len(todo)} images into {OUT}")
    sys.exit(0)

# --- Dataverse: resolve file ids by name, then download each -------------------
print("resolving file ids on Dataverse ...")
with urllib.request.urlopen(f"{API}/datasets/:persistentId/?persistentId={DOI}", timeout=60) as r:
    listing = json.load(r)["data"]["latestVersion"]
lic = listing.get("license", {})
print(f"dataset licence: {lic.get('name')}  {lic.get('uri')}")
by_name = {f["dataFile"]["filename"]: f["dataFile"]["id"] for f in listing["files"]}
missing = [r["filename"] for r in todo if r["filename"] not in by_name]
if missing:
    raise SystemExit(f"not found on Dataverse: {missing}")
for i, r in enumerate(todo, 1):
    fid = by_name[r["filename"]]
    dst = OUT / r["category"] / r["filename"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    with urllib.request.urlopen(f"{API}/access/datafile/{fid}", timeout=120) as resp, \
         open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    tmp.replace(dst)
    print(f"  [{i:2d}/{len(todo)}] {r['filename']}  ({dst.stat().st_size/1e6:.1f} MB)")
print(f"done -> {OUT}")
