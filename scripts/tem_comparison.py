"""
Run the pipeline over the TEM samples and render raw-vs-segmented comparisons.

TEM images here differ from the SEM images the app was built for: there is no
databar, the scale bar is burned into the bottom-left of the frame itself (so it
would be segmented as an object), and features are dark on a bright background.
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from sem_particle_analysis import ParticleAnalyzer, ParticleSegmenter, SAMModel, ScaleDetector
from sem_particle_analysis.utils import load_image

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "figures"
SCALE_REGION = dict(region_x=0.14, region_y=0.955, region_width=0.28, region_height=0.085)

# The burned-in scale bar and its label occupy this corner. It is dark on a
# bright field, so SAM segments it happily; blank it before measuring.
BURNIN_FRAC_W, BURNIN_FRAC_H = 0.30, 0.10

MIN_FG, MAX_FG = 0.0005, 0.35

# TEM features are electron-dense: they appear darker than the support film. A
# candidate mask is only credible if what it encloses is actually darker than
# what it excludes, by at least this many grey levels. Without this, a mask that
# simply covers half the frame of background speckle can win on size alone.
MIN_CONTRAST = 6.0


def mask_burnin(mask):
    out = mask.copy()
    h, w = out.shape
    out[int(h * (1 - BURNIN_FRAC_H)):, : int(w * BURNIN_FRAC_W)] = False
    return out


def pick_mask(segmenter, masks, min_size, image):
    """
    Choose the mask candidate that plausibly represents the dark features.

    Ranked by how much darker the enclosed region is than the rest of the frame.
    Area alone is a bad criterion here: SAM's full-image box prompt happily
    returns a mask covering half the frame of background speckle, which beats a
    thin CNT on every size-based score while containing nothing real.
    """
    grey = image[..., 0].astype(float) if image.ndim == 3 else image.astype(float)
    best = None
    for index in range(len(masks)):
        segmenter.select_mask(index)
        for invert in (True, False):
            candidate = mask_burnin(segmenter.get_binary_mask(invert=invert).astype(bool))
            fraction = float(candidate.mean())
            if not (MIN_FG <= fraction <= MAX_FG):
                continue
            contrast = float(grey[~candidate].mean() - grey[candidate].mean())
            if contrast < MIN_CONTRAST:
                continue
            analyzer = ParticleAnalyzer(min_size=min_size)
            count, _ = analyzer.analyze_mask(candidate, min_size=min_size,
                                             remove_border=False)
            if count == 0:
                continue
            if best is None or contrast > best[0]:
                best = (contrast, index, invert, candidate, analyzer, count)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-sample", type=int, default=4)
    parser.add_argument("--model-type", default="vit_h")
    parser.add_argument("--min-size", type=int, default=30)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = REPO / "sam_weights" / (
        "sam_vit_h_4b8939.pth" if args.model_type == "vit_h" else "sam_vit_b_01ec64.pth")
    sam = SAMModel(str(checkpoint), model_type=args.model_type)
    detector = ScaleDetector(use_gpu=False)

    records = []
    for folder in sorted(glob.glob(str(REPO / "data/raw/*/*/"))):
        folder = Path(folder)
        files = sorted(folder.glob("*.tif"))
        picks = [files[i] for i in np.linspace(0, len(files) - 1, args.per_sample).astype(int)]

        fig, axes = plt.subplots(len(picks), 2, figsize=(11, 5.5 * len(picks)))
        for row, path in enumerate(picks):
            image = load_image(str(path))

            scale_txt, nm_per_px, warn = "not detected", None, None
            try:
                res = detector.detect_scale_bar(image, **SCALE_REGION)
                nm_per_px = res["conversion"]
                warn = res.get("warning")
                scale_txt = (f"{res['scale_nm']:g} nm bar / {res['pixel_length']}px"
                             f" = {nm_per_px:.4g} nm/px")
            except Exception as exc:
                scale_txt = f"scale failed: {str(exc)[:60]}"

            segmenter = ParticleSegmenter(sam)
            masks, _scores = segmenter.segment_image(image, multimask_output=True)
            best = pick_mask(segmenter, masks, args.min_size, image)

            ax_raw, ax_seg = axes[row]
            ax_raw.imshow(image, cmap="gray", vmin=0, vmax=255)
            ax_raw.set_title(f"RAW  {path.name}", fontsize=9)
            ax_raw.add_patch(Rectangle(
                (0, image.shape[0] * (1 - BURNIN_FRAC_H)),
                image.shape[1] * BURNIN_FRAC_W, image.shape[0] * BURNIN_FRAC_H,
                fill=False, edgecolor="orange", lw=1.2, ls="--"))

            ax_seg.imshow(image, cmap="gray", vmin=0, vmax=255)
            if best is None:
                count = 0
                ax_seg.set_title("SEGMENTED — no plausible mask", fontsize=9, color="crimson")
            else:
                contrast, idx, inv, candidate, analyzer, count = best
                overlay = np.zeros((*candidate.shape, 4))
                overlay[candidate] = [1, 0, 0, 0.45]
                ax_seg.imshow(overlay)
                ax_seg.contour(candidate, levels=[0.5], colors="red", linewidths=0.8)
                cov = 100 * candidate.mean()
                ax_seg.set_title(f"SEGMENTED — {count} region(s), {cov:.1f}% of frame, "
                                 f"contrast {contrast:.0f} [mask {idx}, invert={inv}]", fontsize=9)

            note = scale_txt + ("\n⚠ " + warn[:90] if warn else "")
            ax_raw.set_xlabel(note, fontsize=7,
                              color="darkorange" if warn else ("crimson" if nm_per_px is None else "black"))
            for ax in (ax_raw, ax_seg):
                ax.set_xticks([]); ax.set_yticks([])

            records.append({"path": str(path), "sample": folder.name,
                            "nm_per_px": nm_per_px, "scale_text": scale_txt,
                            "warning": warn, "n_regions": count})
            print(f"  {folder.name}/{path.name}: {count} region(s), {scale_txt}")

        fig.suptitle(f"{folder.parent.name}  —  sample {folder.name}", fontsize=12)
        fig.tight_layout()
        dest = OUT / f"comparison_{folder.name}.png"
        fig.savefig(dest, dpi=85, bbox_inches="tight")
        plt.close(fig)
        print(f"-> {dest}")

    (OUT / "comparison_index.json").write_text(json.dumps(records, indent=1))


if __name__ == "__main__":
    main()
