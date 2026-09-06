"""
Figures for the CNT filter study, from the combined reviewed results.

Reads the two tables ``combine.py`` writes — one row per particle, one row per
reviewed frame — and produces the figure set and the summary table.

Three samples, each a filter, each imaged on two instruments:

    A = filter 02      B = filter 10      C = filter 12

Figures 1 to 4 keep SEM and TEM in separate panels. They are not two
measurements of the same thing: at 650–2000x the SEM sees agglomerates lying on
the filter, while the TEM sees individual structures, and their medians differ
by an order of magnitude.

Figure 5 pools them anyway, one distribution per sample, because 34 to 189
structures per sample per instrument is too few to settle a distribution and
the pooled 69 to 270 is better. What that costs is stated on the figure: below
the SEM's resolution limit the curve is TEM alone, so the relative height of
the fine and coarse modes reflects how many frames were taken on each
instrument, not the filter - below 2.15 um, the 200 px floor at the SEM's
finest pixel size, only the TEM contributes at all. Pooling is unweighted; see
figure_combined for why weighting by imaged area cannot be done here.

Sizes are plotted on a log axis throughout. The measured equivalent diameters
span 0.05 to 56 µm — three orders of magnitude — and on a linear axis the whole
TEM distribution collapses into the first bin. The histogram is drawn as
dN/dlog(d) for the same reason: with logarithmic bins, a count per bin makes
the wide bins at the right look fuller than they are, and only the count per
decade is comparable across the axis.

Counts per frame are shown for the SEM alone. The TEM pass segmented the
dominant structure in each frame rather than everything on it, so its roughly
one structure per frame is a fact about the protocol, not about the sample, and
plotting it beside the SEM would invite reading it as loading.

Usage:
    python make_figures.py [data folder] [figure folder]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

DATA = Path(sys.argv[1] if len(sys.argv) > 1
            else r"D:\NIOSH Sample Images\combined")
FIGURES = Path(sys.argv[2] if len(sys.argv) > 2 else DATA / "figures")

# The house style, matching analyze_results.ipynb so these sit beside the
# figures already made for the manuscript.
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
rcParams["font.size"] = 14
rcParams["axes.linewidth"] = 1.5
rcParams["xtick.major.width"] = 1.5
rcParams["ytick.major.width"] = 1.5
rcParams["xtick.major.size"] = 5
rcParams["ytick.major.size"] = 5
rcParams["figure.dpi"] = 150
rcParams["savefig.dpi"] = 300
rcParams["savefig.bbox"] = "tight"
# Plain matplotlib throughout: seaborn is not in the analysis environment, and
# a figure script for a paper is a poor reason to add a dependency to it.
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["axes.grid"] = False

SAMPLES = ["A", "B", "C"]
FILTERS = {"A": "02", "B": "10", "C": "12"}
COLOURS = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}
POSITION_COLOURS = ["#4c72b0", "#dd8452", "#55a868"]
MODALITIES = ["TEM", "SEM"]
POSITIONS = ["Edge", "Middle", "Tip"]
# The two size fractions this work reports: respirable and nano.
THRESHOLDS_UM = {"1 µm": 1.0, "200 nm": 0.2}
# Ticks chosen by hand. Matplotlib's log minor labels collide into each other
# at this figure width — "3 x 10^0" and "4 x 10^0" printed as one word.
TICKS_UM = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]


def label(sample):
    return f"{sample} (filter {FILTERS[sample]})"


def save(fig, name):
    """Both formats, because the manuscript takes PDF and the drafts take PNG."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", facecolor="white")
    fig.savefig(FIGURES / f"{name}.pdf")
    plt.close(fig)
    print(f"  {name}.png / .pdf")


def size_axis(ax, values, which="x"):
    """
    A log size axis labelled in plain numbers rather than powers.

    Formatted with %g so a tick carries only the digits it needs: 0.05 and 50
    rather than 0.05 and 50.00, which is what a shared decimal place does to
    an axis spanning three decades.
    """
    low, high = values.min(), values.max()
    ticks = [t for t in TICKS_UM if low / 1.6 <= t <= high * 1.6]
    axis = ax.xaxis if which == "x" else ax.yaxis
    (ax.set_xscale if which == "x" else ax.set_yscale)("log")
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def spread(ax, groups, colours, positions, log=True):
    """
    A box per group with every point beside it.

    The points are drawn because several of these groups hold a handful of
    particles, and a box plot of eight numbers says more than eight numbers
    can support unless the reader can see there are eight.
    """
    box = ax.boxplot(list(groups), positions=positions, widths=0.55,
                     patch_artist=True, showfliers=False,
                     medianprops=dict(linewidth=2.5, color="black"),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))
    for patch, colour in zip(box["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.35)
        patch.set_edgecolor(colour)
        patch.set_linewidth(1.6)
    rng = np.random.default_rng(0)
    for values, colour, x in zip(groups, colours, positions):
        if not len(values):
            continue
        ax.scatter(x + rng.uniform(-0.18, 0.18, len(values)), values,
                   s=16, alpha=0.65, color=colour, linewidths=0, zorder=3)
    if log:
        ax.set_yscale("log")


def geometric(values):
    """
    Geometric mean and standard deviation.

    The convention for an aerosol size distribution, and the right one here:
    these spread over decades, so the arithmetic mean sits above almost every
    particle in the set.
    """
    logs = np.log(values)
    return float(np.exp(logs.mean())), float(np.exp(logs.std(ddof=1)))


# --- figures -----------------------------------------------------------------

def figure_distributions(particles):
    """Size distribution per sample, one panel per instrument."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    for ax, modality in zip(axes, MODALITIES):
        subset = particles[particles.modality == modality]
        edges = np.linspace(np.log10(subset.equiv_diameter_um.min()),
                            np.log10(subset.equiv_diameter_um.max()), 19)
        width = edges[1] - edges[0]
        for sample in SAMPLES:
            values = subset[subset["sample"] == sample].equiv_diameter_um
            if not len(values):
                continue
            counted, _ = np.histogram(np.log10(values), bins=edges)
            # Fraction of the sample per decade of diameter: the only form in
            # which bars over logarithmic bins can be compared by area.
            density = counted / len(values) / width
            ax.step(10 ** np.repeat(edges, 2)[1:-1],
                    np.repeat(density, 2), where="post",
                    linewidth=2.5, color=COLOURS[sample],
                    label=f"{label(sample)}  n={len(values)}")
            ax.axvline(values.median(), color=COLOURS[sample], linestyle=":",
                       linewidth=1.8, alpha=0.9)
        size_axis(ax, subset.equiv_diameter_um)
        ax.set_xlabel("Equivalent diameter (µm)", fontweight="bold")
        ax.set_ylabel("Fraction per decade", fontweight="bold")
        ax.set_title(modality, fontweight="bold")
        ax.legend(frameon=False, fontsize=11)
    fig.text(0.5, -0.02, "Dotted lines mark the median of each sample.",
             ha="center", fontsize=11, style="italic")
    fig.tight_layout()
    save(fig, "fig1_size_distributions")


def figure_cumulative(particles):
    """
    Cumulative distributions, with the two reported size fractions marked.

    A cumulative plot is what lets a reader take a percentile off the figure,
    and it does not depend on a choice of bin width the way a histogram does.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    for ax, modality in zip(axes, MODALITIES):
        subset = particles[particles.modality == modality]
        for sample in SAMPLES:
            values = np.sort(subset[subset["sample"] == sample].equiv_diameter_um)
            if not len(values):
                continue
            ax.step(values, np.arange(1, len(values) + 1) / len(values),
                    where="post", linewidth=2.5, color=COLOURS[sample],
                    label=f"{label(sample)}  n={len(values)}")
        for text, edge in THRESHOLDS_UM.items():
            if subset.equiv_diameter_um.min() < edge < subset.equiv_diameter_um.max():
                ax.axvline(edge, color="0.35", linestyle="--", linewidth=1.4)
                ax.text(edge, 0.995, f" {text}", ha="left", va="top",
                        fontsize=11, color="0.35", transform=ax.get_xaxis_transform())
        size_axis(ax, subset.equiv_diameter_um)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Equivalent diameter (µm)", fontweight="bold")
        ax.set_ylabel("Cumulative fraction", fontweight="bold")
        ax.set_title(modality, fontweight="bold")
        ax.legend(frameon=False, fontsize=11, loc="lower right")
    fig.tight_layout()
    save(fig, "fig2_cumulative_distributions")


def figure_spread(particles):
    """
    Every particle, by sample and instrument, on one shared size axis.

    Shared deliberately: the gap between what the two instruments resolve is
    itself a result, and separate axes would hide it.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, modality in zip(axes, MODALITIES):
        subset = particles[particles.modality == modality]
        order = [s for s in SAMPLES if (subset["sample"] == s).any()]
        groups = [subset[subset["sample"] == s].equiv_diameter_um.values
                  for s in order]
        spread(ax, groups, [COLOURS[s] for s in order], list(range(len(order))))
        for i, values in enumerate(groups):
            # Axes coordinates, so the label cannot land on the title however
            # far the shared axis reaches.
            ax.text(i, 0.99, f"n={len(values)}", ha="center", va="top",
                    fontsize=11, transform=ax.get_xaxis_transform())
        ax.set_xlabel("Sample", fontweight="bold")
        ax.set_ylabel("Equivalent diameter (µm)", fontweight="bold")
        ax.set_title(modality, fontweight="bold")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([label(s) for s in order], fontsize=11)
    size_axis(axes[0], particles.equiv_diameter_um, which="y")
    fig.tight_layout()
    save(fig, "fig3_size_by_sample")


def figure_loading(frames):
    """
    How much was on each SEM frame, and where on the filter it came from.

    Every reviewed frame counts, including the ones with nothing on them: a
    frame an analyst looked at and found empty is a measurement of the sample,
    and leaving it out would put every mean up.

    SEM only. The TEM pass took the dominant structure per frame, so a TEM
    count per frame measures the protocol rather than the filter.
    """
    sem = frames[frames.modality == "SEM"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

    ax = axes[0]
    order = [s for s in SAMPLES if (sem["sample"] == s).any()]
    groups = [sem[sem["sample"] == s].n_particles.values for s in order]
    spread(ax, groups, [COLOURS[s] for s in order], list(range(len(order))),
           log=False)
    ax.set_ylim(0, max(1, sem.n_particles.max()) * 1.12)
    for i, s_name in enumerate(order):
        part = sem[sem["sample"] == s_name]
        ax.text(i, -0.12, f"{len(part)} frames, {int((part.n_particles == 0).sum())} empty",
                ha="center", va="top", fontsize=10, color="0.3",
                transform=ax.get_xaxis_transform())
    ax.set_xlabel("Sample", fontweight="bold", labelpad=34)
    ax.set_ylabel("Structures per frame", fontweight="bold")
    ax.set_title("Loading by sample", fontweight="bold")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([label(s) for s in order], fontsize=11)

    ax = axes[1]
    grouped = sem.groupby(["sample", "position"]).n_particles
    means = grouped.mean().unstack().reindex(index=order, columns=POSITIONS)
    counts = grouped.size().unstack().reindex(index=order, columns=POSITIONS)
    width = 0.26
    for j, position in enumerate(POSITIONS):
        x = np.arange(len(means)) + (j - 1) * width
        ax.bar(x, means[position].values, width, label=position,
               edgecolor="black", linewidth=1.2, color=POSITION_COLOURS[j])
        for xi, value, n in zip(x, means[position].values, counts[position].values):
            if np.isfinite(value):
                ax.text(xi, value + 0.15, f"{int(n)}", ha="center", fontsize=9,
                        color="0.3")
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels([label(s) for s in means.index], fontsize=11)
    ax.set_xlabel("Sample", fontweight="bold")
    ax.set_ylabel("Mean structures per frame", fontweight="bold")
    ax.set_title("Loading by position on the filter", fontweight="bold")
    ax.legend(title="", frameon=False, fontsize=11)
    fig.text(0.5, -0.04, "SEM only — the TEM pass segmented one dominant "
                         "structure per frame, so its per-frame count is not a "
                         "measure of loading.  Numbers above the bars are frames.",
             ha="center", fontsize=11, style="italic")
    fig.tight_layout()
    save(fig, "fig4_sem_loading")


def figure_combined(particles):
    """
    Both instruments in one distribution per sample, size profile and
    cumulative, in the style of the group's other size figures.

    Combined unweighted, which is the only defensible way to pool these two.
    The obvious alternative - weight each particle by the area its instrument
    imaged, so the pool reflects the filter rather than how many frames were
    taken of each - assumes both instruments surveyed a fixed field. The SEM
    did. The TEM did not: its magnification runs over 200-fold across frames
    and correlates 0.82 with the size of the structure in the frame, because
    the operator zoomed to suit each object. Areas from that would be
    meaningless, so what is pooled here is the measured structures, and the
    curve is a size profile of what was measured rather than a
    concentration-weighted population.

    The density is empirical, not a fitted lognormal. A single lognormal is
    rejected for two of the three samples, which is what a distribution built
    from a fine mode the TEM resolves and a coarse mode the SEM surveys should
    do; drawing a smooth fit over that would describe neither mode.
    """
    from matplotlib.lines import Line2D
    from scipy import stats

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4))
    grid = np.logspace(np.log10(particles.equiv_diameter_um.min() / 1.4),
                       np.log10(particles.equiv_diameter_um.max() * 1.4), 400)

    entries = []
    for sample in SAMPLES:
        values = particles[particles["sample"] == sample].equiv_diameter_um.values
        if not len(values):
            continue
        median = float(np.median(values))
        density = stats.gaussian_kde(np.log(values), bw_method=0.35)(np.log(grid))
        # No median line on the density panel. A vertical at the median of a
        # curve whose peak is somewhere else reads as a feature of the curve
        # and is not one; on the cumulative panel the same line lands on the
        # 0.5 crossing and explains itself.
        axes[0].plot(grid, density, linewidth=2.8, color=COLOURS[sample])

        ordered = np.sort(values)
        axes[1].plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered),
                     linewidth=2.8, color=COLOURS[sample])
        axes[1].plot([median, median], [0, 0.5], color=COLOURS[sample],
                     linestyle="--", linewidth=1.8)
        entries.append((sample, len(values), median))

    axes[1].axhline(0.5, color="black", linewidth=1.0)

    for ax, letter, title, ylabel in (
            (axes[0], "A", "Size distributions (d$N$/dln$d$)",
             "Normalised d$N$/dln$d$"),
            (axes[1], "B", "Cumulative particle size distributions",
             "Cumulative probability")):
        size_axis(ax, particles.equiv_diameter_um)
        ax.set_xlim(grid[0], grid[-1])
        ax.set_xlabel("Particle equivalent diameter (µm, log scale)",
                      fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title, fontsize=15)
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.text(-0.10, 1.06, letter, transform=ax.transAxes, fontsize=20,
                fontweight="bold", va="top")

    axes[0].set_ylim(0, None)
    axes[1].set_ylim(0, 1.0)
    # One legend, under both panels and outside them. Placed inside, it either
    # sat over the fine mode on the left or the foot of the curves on the
    # right, and the two panels label the same three samples anyway.
    fig.legend(
        [Line2D([0], [0], color=COLOURS[s], linewidth=2.8) for s, _n, _m in entries],
        [f"{label(s)}   n={n}   CMD={median:.2f} µm" for s, n, median in entries],
        loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=len(entries),
        frameon=False, fontsize=12.5, columnspacing=3.0, handlelength=2.4)
    fig.text(0.5, -0.075,
             "TEM and SEM structures pooled per sample. Dashed lines in B mark "
             "each sample's count median diameter.",
             ha="center", fontsize=11, style="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig5_combined_size_distribution")


def summary(particles, frames):
    """The numbers behind the figures, as a table and as a picture of one."""
    rows = []
    # "Combined" last, as the pooled row figure 5 is drawn from.
    for modality in MODALITIES + ["Combined"]:
        for sample in SAMPLES:
            chosen = (particles["sample"] == sample)
            in_frames = (frames["sample"] == sample)
            if modality != "Combined":
                chosen &= particles.modality == modality
                in_frames &= frames.modality == modality
            p = particles[chosen]
            f = frames[in_frames]
            if not len(p):
                continue
            d = p.equiv_diameter_um.values
            gm, gsd = geometric(d)
            rows.append({
                "Instrument": modality, "Sample": label(sample),
                "Images": len(f), "Empty images": int((f.n_particles == 0).sum()),
                "Structures": len(d),
                "Median (µm)": np.median(d), "GM (µm)": gm, "GSD": gsd,
                "Mean (µm)": d.mean(),
                "P10 (µm)": np.percentile(d, 10), "P90 (µm)": np.percentile(d, 90),
                "Min (µm)": d.min(), "Max (µm)": d.max(),
                "Under 1 µm": int((d < 1.0).sum()),
                "Under 200 nm": int((d < 0.2).sum()),
            })
    table = pd.DataFrame(rows)
    table.to_csv(FIGURES / "summary_statistics.csv", index=False)

    shown = table.copy()
    for column in shown.columns:
        # "Under 1 µm" is a count of particles, not a length: matching on the
        # unit alone printed it as 40.000.
        if column.startswith("Under") or "µm" not in column and column != "GSD":
            continue
        shown[column] = shown[column].map(lambda v: f"{v:.2f}")
    fig, ax = plt.subplots(figsize=(18, 1.1 + 0.42 * len(shown)))
    ax.axis("off")
    drawn = ax.table(cellText=shown.values, colLabels=shown.columns,
                     cellLoc="center", loc="center")
    drawn.auto_set_font_size(False)
    drawn.set_fontsize(10)
    drawn.scale(1, 1.6)
    for i in range(len(shown.columns)):
        drawn[0, i].set_facecolor("#40466e")
        drawn[0, i].set_text_props(color="white", fontweight="bold")
    save(fig, "table1_summary_statistics")
    return table


def main():
    particles = pd.read_csv(DATA / "particles.csv")
    frames = pd.read_csv(DATA / "frames.csv")
    print(f"{len(particles)} particles, {len(frames)} frames\n")

    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_distributions(particles)
    figure_cumulative(particles)
    figure_spread(particles)
    figure_loading(frames)
    figure_combined(particles)
    table = summary(particles, frames)

    print()
    print(table.round(3).to_string(index=False))
    print(f"\nfigures in {FIGURES}")


if __name__ == "__main__":
    main()
