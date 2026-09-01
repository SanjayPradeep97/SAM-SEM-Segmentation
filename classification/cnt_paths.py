"""cnt_paths.py -- where the data lives.  Nothing in this repository hard-codes
a drive letter; every script resolves its inputs through this module.

    CNT_BASE      (required for anything that touches images or feature caches)
                  Root directory that contains
                      NIOSH Dataset/CNT-Fiber/CNT-Fiber-0001.tif ...   (Dataverse)
                      NIOSH Dataset/Masks/masks/CNT-Fiber-0001_mask.png ...
                  Feature caches are written to  <CNT_BASE>/Encoder Benchmark/
                  and download caches to         <CNT_BASE>/_cache/.
    CNT_MASKS     (optional) directory of <base>_mask.png files, if the masks
                  are kept somewhere other than <CNT_BASE>/NIOSH Dataset/Masks/masks.
    CNT_RESULTS   (optional) results directory; default is <repo>/results.
    CNT_SPLITS    (optional) split file; default is <repo>/splits/dataset_splits.pkl.

The split file stores the absolute paths of the machine it was created on.
Those are never used: `image_path()` rebuilds every path from the category
and file name, so the same pickle works on any machine.
"""
from __future__ import annotations
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def base(required=True) -> Path | None:
    v = os.environ.get("CNT_BASE")
    if not v:
        if not required:
            return None
        raise SystemExit(
            "CNT_BASE is not set.\n"
            "  Point it at the directory that holds 'NIOSH Dataset/' (the Harvard\n"
            "  Dataverse download, https://doi.org/10.7910/DVN/5O0SF7) and the masks.\n"
            "  Feature caches and download caches are created underneath it.\n"
            "      set CNT_BASE=D:\\path\\to\\data         (cmd)\n"
            "      export CNT_BASE=/path/to/data          (bash)")
    p = Path(v)
    if required and not p.is_dir():
        raise SystemExit(f"CNT_BASE={v} is not a directory")
    return p


def results_dir() -> Path:
    return Path(os.environ.get("CNT_RESULTS") or REPO / "results")


def splits_file() -> Path:
    return Path(os.environ.get("CNT_SPLITS") or REPO / "splits" / "dataset_splits.pkl")


def bench_dir() -> Path:
    """Feature caches (feats_*.npz).  ~1.5 GB for the full roster."""
    return base() / "Encoder Benchmark"


def images_dir() -> Path:
    return base() / "NIOSH Dataset"


def masks_dir() -> Path:
    v = os.environ.get("CNT_MASKS")
    return Path(v) if v else images_dir() / "Masks" / "masks"


def image_path(category: str, filename: str) -> Path:
    return images_dir() / f"CNT-{category}" / filename


def mask_path(filename: str) -> Path:
    return masks_dir() / f"{Path(filename).stem}_mask.png"
