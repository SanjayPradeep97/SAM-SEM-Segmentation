"""
cache_setup.py -- pin every cache to a stable location so nothing is fetched or
recomputed twice.

IMPORT THIS FIRST, before timm / transformers / ultralytics.  HuggingFace reads
its cache location at import time, so setting HF_HOME afterwards has no effect.

Four things were being redone on every run:

  1. timm / HF encoder weights   ~/.cache by default, and re-resolved over the
                                 network on each import unless pinned.
  2. Ultralytics .pt checkpoints downloaded into the CURRENT directory, so
                                 running from anywhere else re-downloads.
  3. The AMP-check model         yolo26n.pt, fetched into the current directory.
  4. Decoded images              2.4 GB of 1350x1040 TIFFs re-decoded for every
                                 (split, variant) on every process start.

(4) is by far the largest: the smoke run spent most of its eleven minutes
decoding, not training.
"""
from __future__ import annotations
import os
from pathlib import Path


def root(base=None) -> Path:
    v = base or os.environ.get("CNT_BASE")
    if not v:
        raise SystemExit("cache_setup: CNT_BASE is not set (see classification/cnt_paths.py)")
    b = Path(v)
    d = b / "_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def setup(base=None, verbose=True):
    """Point every downloader at <CNT_BASE>/_cache. Safe to call more than once."""
    c = root(base)
    for sub in ("hf", "torch", "weights", "prep", "ultralytics"):
        (c / sub).mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(c / "hf"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(c / "hf" / "hub"))
    os.environ.setdefault("TORCH_HOME", str(c / "torch"))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(c / "ultralytics"))
    # quieten the unauthenticated-HF-Hub warning; it is not an error
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if verbose:
        print(f"  caches -> {c}", flush=True)
    return c


def weights_dir(base=None) -> Path:
    return root(base) / "weights"


def yolo_checkpoint(name, base=None, verbose=True):
    """Absolute path to a YOLO checkpoint, downloading once into the cache.

    Ultralytics resolves a bare filename against the CURRENT directory and
    downloads there if absent, so a bare name re-downloads whenever the run
    starts from a different folder.  An absolute path is stable.
    """
    w = weights_dir(base)
    p = w / name
    if p.exists():
        if verbose:
            print(f"  [cache] {name} already present", flush=True)
        return str(p)
    import shutil, urllib.request
    url = ("https://github.com/ultralytics/assets/releases/download/v8.4.0/" + name)
    tmp = p.with_suffix(p.suffix + ".part")
    try:
        if verbose:
            print(f"  [cache] downloading {name} -> {w}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
            shutil.copyfileobj(r, fh)
        tmp.replace(p)
        return str(p)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        if verbose:
            print(f"  [cache] could not pre-fetch {name} ({type(e).__name__}); "
                  f"letting ultralytics resolve it", flush=True)
        return name          # fall back to ultralytics' own resolution


def prefetch_amp_model(base=None, verbose=True):
    """Ultralytics' AMP check loads yolo26n.pt from the CURRENT directory.

    Put a copy there once so the check does not re-download per run.
    """
    src = yolo_checkpoint("yolo26n.pt", base, verbose=False)
    dst = Path.cwd() / "yolo26n.pt"
    if Path(src).is_absolute() and Path(src).exists() and not dst.exists():
        import shutil
        shutil.copy2(src, dst)
        if verbose:
            print(f"  [cache] AMP check model staged in {dst.parent}", flush=True)


# ---------------------------------------------------------------------------
def prep_cache_path(tag, use_mask, rows, max_side, base=None):
    """Content-addressed: the key covers every path, the order, and max_side, so
    a changed split or resolution can never silently reuse a stale cache."""
    import hashlib
    h = hashlib.sha256()
    h.update(f"{tag}|{use_mask}|{max_side}|{len(rows)}".encode())
    for r in rows:
        h.update(str(r["image"]).encode()); h.update(b"\0")
        h.update(str(r.get("mask", "")).encode()); h.update(b"\0")
        h.update(str(r["y"]).encode()); h.update(b"\1")
    return root(base) / "prep" / f"prep_{tag}_{'m' if use_mask else 'r'}_{h.hexdigest()[:16]}.npz"
