"""OPTIONAL: fetch the original TIFFs of the 60 demo micrographs.

The demo does not need this.  demo/images/ already contains the same 60
images as lossless PNGs (see ATTRIBUTION.md).  This script exists for anyone
who wants the untouched originals from the NIOSH TEM dataset on Harvard
Dataverse (doi:10.7910/DVN/5O0SF7, CC BY-NC 4.0), or wants to confirm that
the shipped PNGs are pixel-identical to them.  It fetches exactly the files
listed in demo_images.csv, either

  * from a local copy of the dataset (set CNT_BASE, see classification/cnt_paths.py)
        python fetch_demo_images.py
  * or straight from Dataverse, one file at a time (about 85 MB in total)
        python fetch_demo_images.py --download

The TIFFs land next to the PNGs (demo/images/<class>/<name>.tif; git ignores
them).  Nothing else in the dataset is touched.

        python fetch_demo_images.py --download --verify

additionally compares every TIFF with the shipped PNG, pixel for pixel, and
exits non-zero on any difference.
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


def fetch_local(todo):
    if cnt_paths.base(required=False) is None:
        raise SystemExit("CNT_BASE is not set.  Either point it at a local copy of the\n"
                         "dataset, or run with --download to fetch the files from Dataverse.")
    for r in todo:
        src = cnt_paths.image_path(r["category"], r["filename"])
        if not src.exists():
            raise SystemExit(f"missing: {src}")
        dst = OUT / r["category"] / r["filename"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print(f"copied {len(todo)} images into {OUT}")


def _open(url, timeout):
    # Dataverse redirects file downloads to S3, which refuses urllib's default
    # User-Agent with a 403; any browser-like agent string is accepted.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (fetch_demo_images.py)"})
    return urllib.request.urlopen(req, timeout=timeout)


def fetch_dataverse(todo):
    print("resolving file ids on Dataverse ...")
    with _open(f"{API}/datasets/:persistentId/?persistentId={DOI}", 60) as r:
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
        with _open(f"{API}/access/datafile/{fid}", 120) as resp, \
             open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
        tmp.replace(dst)
        print(f"  [{i:2d}/{len(todo)}] {r['filename']}  ({dst.stat().st_size/1e6:.1f} MB)")
    print(f"done -> {OUT}")


def verify(rows):
    import numpy as np
    from PIL import Image
    bad = 0
    for r in rows:
        stem = Path(r["filename"]).stem
        t = np.asarray(Image.open(OUT / r["category"] / r["filename"]).convert("L"))
        g = np.asarray(Image.open(OUT / r["category"] / f"{stem}.png"))
        if not np.array_equal(t, g):
            bad += 1
            print(f"  DIFFERS: {r['filename']}")
    print(f"verified {len(rows) - bad}/{len(rows)} shipped PNGs identical to the original TIFFs")
    return bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true",
                    help="fetch from Harvard Dataverse instead of a local copy")
    ap.add_argument("--verify", action="store_true",
                    help="compare each TIFF with the shipped PNG, pixel for pixel")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(MANIFEST)))
    todo = [r for r in rows if not (OUT / r["category"] / r["filename"]).exists()]
    print(f"{len(rows)} images in the manifest, {len(rows) - len(todo)} TIFFs already present")
    if todo:
        (fetch_dataverse if a.download else fetch_local)(todo)
    if a.verify:
        return 0 if verify(rows) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
