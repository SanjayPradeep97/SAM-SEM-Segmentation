"""
Headless batch analysis.

Runs the same pipeline as the web app over a folder of micrographs without any
clicking, and records enough about each run to reproduce it later: the exact
inputs, the model, the scale and where it came from, every parameter, and the
library versions. Interactive refinement is deliberately not available here —
anything this tool reports came out of the automatic pipeline alone.

    sem-analyze data/raw/sample-a -o data/processed/sample-a
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .analysis import ParticleAnalyzer
from .model import SAMModel, discover_checkpoints, infer_model_type
from .scale_detection import ScaleDetector
from .segmentation import ParticleSegmenter
from .utils import find_images_in_folder, load_image

# Used only when the databar can't be measured.
DEFAULT_CROP_PERCENT = 7.0

def _repo_root():
    # cli.py -> sem_particle_analysis -> sem_particle_analysis -> repo root
    return Path(__file__).resolve().parents[2]


def find_checkpoint(model_type):
    """Locate SAM weights for ``model_type``, or return None."""
    return next((p for p in discover_checkpoints()
                 if infer_model_type(p, default=None) == model_type), None)


def sha256_file(path, chunk_size=1 << 20):
    """SHA-256 of a file, read in chunks so multi-GB checkpoints don't blow up RAM."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision():
    try:
        out = subprocess.run(
            ["git", "-C", str(_repo_root()), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _versions():
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("numpy", "pandas", "torch", "skimage", "cv2", "easyocr"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = None
    try:
        from . import __version__

        versions["sem_particle_analysis"] = __version__
    except Exception:
        versions["sem_particle_analysis"] = None
    return versions


def resolve_scale(detector, image, image_path, args):
    """
    Determine nm/pixel for one image.

    Returns:
        tuple: (nm_per_px, provenance_dict). nm_per_px is None when scale could
        not be established, in which case measurements stay in pixels.
    """
    if args.scale_nm_per_px is not None:
        return args.scale_nm_per_px, {"method": "manual", "nm_per_px": args.scale_nm_per_px}

    try:
        result = detector.detect_scale(image, file_path=str(image_path), method=args.scale_method)
        return result["conversion"], {
            "method": result.get("method", args.scale_method),
            "nm_per_px": result["conversion"],
            "scale_nm": result.get("scale_nm"),
            "pixel_length": result.get("pixel_length"),
            "ocr_text": result.get("ocr_text"),
            "warning": result.get("warning"),
        }
    except Exception as exc:
        return None, {"method": "failed", "nm_per_px": None, "error": str(exc)}


def crop_databar(detector, image, args):
    """
    Remove the instrument databar from the bottom of the frame.

    With --crop-percent left at auto, the databar's height is measured. A fixed
    percentage is fragile in both directions: too small leaves a strip whose text
    and borders segment into spurious particles, too large eats real image area.

    Returns:
        tuple: (cropped_image, info_dict)
    """
    height = image.shape[0]

    if args.crop_percent is not None:
        if args.crop_percent <= 0:
            return image, {"method": "none", "rows_removed": 0}
        cropped = detector.crop_scale_bar(image, crop_percent=args.crop_percent)
        return cropped, {"method": "fixed-percent", "percent": args.crop_percent,
                         "rows_removed": height - cropped.shape[0]}

    try:
        databar = detector.detect_databar(image)
    except Exception:
        databar = {}

    if databar.get("has_databar") and databar.get("databar_height"):
        keep = height - int(databar["databar_height"])
        if 0 < keep < height:
            return image[:keep].copy(), {
                "method": "detected",
                "rows_removed": height - keep,
                "fraction": round(databar.get("databar_fraction", 0.0), 4),
            }

    cropped = detector.crop_scale_bar(image, crop_percent=DEFAULT_CROP_PERCENT)
    return cropped, {"method": "fallback-percent", "percent": DEFAULT_CROP_PERCENT,
                     "rows_removed": height - cropped.shape[0]}


def analyze_image(image_path, sam_model, detector, args):
    """Run scale detection, segmentation and measurement for a single image."""
    image = load_image(str(image_path))

    nm_per_px, scale_info = resolve_scale(detector, image, image_path, args)

    # Trim the databar so it can't be segmented as a particle. Measuring its
    # height beats a fixed percentage, which either leaves a strip behind (and
    # the leftover text fragments into "particles") or eats into the micrograph.
    working, crop_info = crop_databar(detector, image, args)

    segmenter = ParticleSegmenter(sam_model)
    masks, scores = segmenter.segment_image(working, multimask_output=True)

    # Same ranking the app shows the analyst, so a batch run and an interactive
    # one agree about which mask is the right one.
    candidates = segmenter.rank_candidates(working, masks, top_k=3)
    if not candidates:
        raise ValueError(
            "No mask candidate isolated anything convincing; the image may need "
            "interactive segmentation."
        )

    chosen = candidates[0]
    analyzer = ParticleAnalyzer(conversion_factor=nm_per_px, min_size=args.min_size)
    analyzer.analyze_mask(
        chosen["mask"], min_size=args.min_size,
        remove_border=True, border_buffer=args.border_buffer,
    )
    mask_index, inverted, fraction = (
        chosen["mask_index"], chosen["inverted"], chosen["fraction"])
    rejected = [{k: c[k] for k in ("mask_index", "inverted", "fraction", "contrast")}
                for c in candidates[1:]]

    if args.clear_edges:
        analyzer.clear_edge_particles(buffer_size=0)

    measurements = analyzer.get_measurements(in_nm=nm_per_px is not None)
    stats = analyzer.get_summary_statistics()

    return {
        "measurements": measurements,
        "stats": stats,
        "scale": scale_info,
        "crop": crop_info,
        "mask_index": int(mask_index),
        "mask_inverted": bool(inverted),
        "mask_foreground_fraction": round(fraction, 4),
        "mask_scores": [float(s) for s in scores],
        "other_candidates": rejected,
        "image_shape": list(working.shape[:2]),
    }


def write_plots(per_particle, out_dir, unit):
    """Publication-ready size distributions at 300 dpi, in PNG and PDF."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if per_particle.empty:
        return []

    written = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(per_particle["diameter"], bins=25, color="#4C72B0", edgecolor="white")
    axes[0].set_xlabel(f"Equivalent diameter ({unit})")
    axes[0].set_ylabel("Count")
    axes[1].hist(per_particle["area"], bins=25, color="#DD8452", edgecolor="white")
    axes[1].set_xlabel(f"Area ({unit}²)")
    axes[1].set_ylabel("Count")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Particle size distribution (n = {len(per_particle)})")
    fig.tight_layout()

    for suffix in ("png", "pdf"):
        path = out_dir / f"size_distribution.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        written.append(path.name)
    plt.close(fig)
    return written


def build_parser():
    parser = argparse.ArgumentParser(
        prog="sem-analyze",
        description="Batch particle analysis for SEM/TEM micrographs (no GUI).",
    )
    parser.add_argument("input", type=Path,
                        help="Image file, or folder of images to process")
    parser.add_argument("-o", "--output", type=Path, default=Path("analysis_output"),
                        help="Directory for CSVs, plots and the provenance record")

    model = parser.add_argument_group("model")
    model.add_argument("--model-type", default="vit_h",
                       choices=["vit_b", "vit_h", "vit_l"])
    model.add_argument("--checkpoint", type=Path,
                       help="SAM weights (default: look in sam_weights/)")
    model.add_argument("--device", help="Force a device, e.g. cpu, mps, cuda")

    scale = parser.add_argument_group("scale")
    scale.add_argument("--scale-method", default="auto",
                       choices=["auto", "metadata", "ocr"],
                       help="How to establish nm/pixel (default: auto)")
    scale.add_argument("--scale-nm-per-px", type=float,
                       help="Override detection with a fixed nm/pixel for every image")
    scale.add_argument("--crop-percent", type=float, default=None,
                       help="Percent of image height to trim off the bottom. "
                            "Default is to measure the databar; use 0 to keep the "
                            "full frame")

    analysis = parser.add_argument_group("analysis")
    analysis.add_argument("--min-size", type=int, default=30,
                          help="Discard particles smaller than this many pixels (default: 30)")
    analysis.add_argument("--border-buffer", type=int, default=4)
    analysis.add_argument("--clear-edges", action="store_true",
                          help="Drop particles touching the frame edge; they are only "
                               "partly imaged, so their size is not a real measurement")

    output = parser.add_argument_group("output")
    output.add_argument("--no-plots", action="store_true")
    output.add_argument("--no-checkpoint-hash", action="store_true",
                        help="Skip hashing the weights file (saves a few seconds)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.input.is_dir():
        image_paths = [Path(p) for p in find_images_in_folder(str(args.input))]
    elif args.input.exists():
        image_paths = [args.input]
    else:
        print(f"error: no such file or folder: {args.input}", file=sys.stderr)
        return 2

    if not image_paths:
        print(f"error: no images found in {args.input}", file=sys.stderr)
        return 2

    checkpoint = args.checkpoint or find_checkpoint(args.model_type)
    if checkpoint is None or not Path(checkpoint).exists():
        print(
            f"error: no {args.model_type} weights found. Pass --checkpoint, or run:\n"
            f"    python download_sam_weights.py",
            file=sys.stderr,
        )
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Analysing {len(image_paths)} image(s) -> {args.output}")

    sam_model = SAMModel(str(checkpoint), model_type=args.model_type, device=args.device)
    detector = ScaleDetector(use_gpu=False)

    per_particle_rows = []
    per_image_rows = []
    image_records = []

    for number, image_path in enumerate(image_paths, start=1):
        print(f"[{number}/{len(image_paths)}] {image_path.name}")
        try:
            result = analyze_image(image_path, sam_model, detector, args)
        except Exception as exc:
            print(f"    failed: {exc}", file=sys.stderr)
            image_records.append({"image": str(image_path), "error": str(exc)})
            continue

        measurements = result["measurements"]
        stats = result["stats"]
        unit = measurements["unit"]
        scale_warning = result["scale"].get("warning")

        for index, (area, diameter, centroid) in enumerate(
            zip(measurements["areas"], measurements["diameters"], measurements["centroids"]), start=1
        ):
            per_particle_rows.append({
                "image": image_path.name,
                "particle": index,
                "area": area,
                "diameter": diameter,
                "centroid_x": centroid[0],
                "centroid_y": centroid[1],
                "unit": unit,
            })

        per_image_rows.append({
            "image": image_path.name,
            "num_particles": stats.get("num_particles", 0),
            "nm_per_px": result["scale"]["nm_per_px"],
            "scale_method": result["scale"]["method"],
            "area_mean": stats.get("area_mean"),
            "area_median": stats.get("area_median"),
            "area_std": stats.get("area_std"),
            "diameter_mean": stats.get("diameter_mean"),
            "diameter_median": stats.get("diameter_median"),
            "diameter_std": stats.get("diameter_std"),
            "unit": unit,
        })

        image_records.append({
            "image": str(image_path.resolve()),
            "sha256": sha256_file(image_path),
            "shape": result["image_shape"],
            "scale": result["scale"],
            "crop": result["crop"],
            "mask_index": result["mask_index"],
            "mask_inverted": result["mask_inverted"],
            "mask_foreground_fraction": result["mask_foreground_fraction"],
            "mask_scores": result["mask_scores"],
            "other_candidates": result["other_candidates"],
            "num_particles": stats.get("num_particles", 0),
        })

        note = f"  ⚠️  {scale_warning}" if scale_warning else ""
        print(f"    {stats.get('num_particles', 0)} particles, "
              f"scale={result['scale']['method']}{note}")

    per_particle = pd.DataFrame(per_particle_rows)
    per_image = pd.DataFrame(per_image_rows)
    per_particle.to_csv(args.output / "particles.csv", index=False)
    per_image.to_csv(args.output / "per_image_summary.csv", index=False)

    plots = []
    if not args.no_plots and not per_particle.empty:
        unit = per_particle["unit"].iloc[0]
        plots = write_plots(per_particle, args.output, unit)

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "git_revision": _git_revision(),
        "versions": _versions(),
        "device": str(sam_model.device),
        "model": {
            "type": args.model_type,
            "checkpoint": str(Path(checkpoint).resolve()),
            "sha256": None if args.no_checkpoint_hash else sha256_file(checkpoint),
        },
        "parameters": {
            "min_size": args.min_size,
            "border_buffer": args.border_buffer,
            "clear_edges": args.clear_edges,
            "crop_percent": args.crop_percent,
            "scale_method": args.scale_method,
            "scale_nm_per_px": args.scale_nm_per_px,
        },
        "images": image_records,
        "outputs": ["particles.csv", "per_image_summary.csv", *plots],
        "note": "Automatic pipeline only — no interactive refinement was applied.",
    }
    with open(args.output / "run.json", "w") as handle:
        json.dump(provenance, handle, indent=2)

    failed = sum(1 for record in image_records if "error" in record)
    print(f"\nDone. {len(per_image)} image(s) analysed"
          + (f", {failed} failed" if failed else "")
          + f", {len(per_particle)} particles total.")
    print(f"Wrote {args.output}/particles.csv, per_image_summary.csv, run.json"
          + (f", {', '.join(plots)}" if plots else ""))
    return 1 if failed and per_image.empty else 0


if __name__ == "__main__":
    raise SystemExit(main())
