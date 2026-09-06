"""
Audit every SEM scale, against sources that do not share a failure.

The SEM frames took their pixel size from the instrument's own record rather
than from OCR, so the usual worry - a misread glyph - does not apply. What can
still be wrong is subtler: the wrong metadata field read, the databar counted
as image, a frame carrying a scale that belongs to a different magnification.
Those show up as disagreement between independent records, so this compares
four:

  PixelWidth    what the analysis used, re-read here from the file itself
                rather than taken from the results CSV;
  HFW / ResX    the field width over the scan width - the same quantity
                reached through different fields;
  magnification 150x is 1798.61 nm/px on this column and every other
                magnification is that scaled by the ratio, so the printed
                magnification predicts the pixel size outright;
  the printed bar  measured from the pixels of the databar and divided into
                its printed label.

The bar is measured from the tick row: the topmost bright rows of the databar
hold the two end ticks and nothing else, so no label text can stretch the span.

On the label. OCR reads the micron sign as a p about as often as not - "100 pm"
for "100 um" - and a unit taken from that would be wrong by a factor of a
million. So the digits are taken from OCR and the unit is chosen as whichever
of nm/um/mm brings digits/span within five percent of the recorded scale. That
still tests what matters: the digits and the span are both independent of the
metadata, and only an error of exactly a thousand could be absorbed by the
choice of unit. Anything else fails, and every crop is captioned for a person
to read the label with their own eyes.
"""
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

warnings.filterwarnings("ignore")

SOURCE = Path(r"D:\NIOSH Sample Images\SEM Images")
ANALYSIS = Path(r"D:\NIOSH Sample Images\SEM Images\SEM_analysis")
OUT = Path(sys.argv[1])

REFERENCE_MAG, REFERENCE_NM = 150.0, 1798.61
UNITS = {"nm": 1.0, "um": 1e3, "mm": 1e6}
# 1, 2, 3 and 5 times a power of ten: what an instrument prints on a bar.
STANDARD = [v * 10 ** p for p in range(0, 8) for v in (1, 2, 3, 5)]
TOLERANCE = 0.05


def metadata(path):
    """The scale fields FEI writes into tag 34682."""
    text = Image.open(path).tag_v2[34682]

    def field(name):
        hit = re.search(rf"^{name}=(.+)$", text, re.M)
        return hit.group(1).strip() if hit else None

    return {
        "pixel_width_m": float(field("PixelWidth")),
        "pixel_height_m": float(field("PixelHeight")),
        "hfw_m": float(field("HFW")),
        "res_x": int(field("ResolutionX")),
        "res_y": int(field("ResolutionY")),
        # FEI stores no magnification number; the databar prints one, and it
        # is the canvas width over the field width.
        "magnification": float(field("MagCanvasRealWidth")) / float(field("HFW")),
        "hv_kv": float(field("HV")) / 1000.0,
    }


def bar_span(strip):
    """
    The printed bar's length in pixels, measured from its two end ticks.

    Returns:
        tuple: (x0, x1, length) or None if no tick pair is found.
    """
    bright = strip > 200
    for row in range(strip.shape[0]):
        columns = np.where(bright[row])[0]
        if columns.size < 2:
            continue
        groups = np.split(columns, np.where(np.diff(columns) > 3)[0] + 1)
        if len(groups) != 2 or any(len(g) > 6 for g in groups):
            continue
        x0, x1 = int(groups[0].min()), int(groups[1].max())
        if x1 - x0 < 100:
            continue
        return x0, x1, x1 - x0 + 1
    return None


def read_digits(reader, strip, x0, x1):
    """The number printed on the bar, and the raw text it was read from."""
    patch = strip[:, max(0, x0 - 4):min(strip.shape[1], x1 + 5)]
    big = Image.fromarray(patch).resize(
        (patch.shape[1] * 2, patch.shape[0] * 2), Image.LANCZOS)
    text = " ".join(t for _box, t, _c in reader.readtext(np.array(big)))
    hit = re.search(r"(\d[\d\s]{0,5})\s*[a-zµμ]", text)
    if not hit:
        hit = re.search(r"(\d[\d\s]{0,5})", text)
    if not hit:
        return None, text
    return int(hit.group(1).replace(" ", "")), text


def unit_that_fits(digits, span, stored):
    """
    Which of nm, um, mm makes the printed bar agree with the recorded scale.

    Returns:
        tuple: (unit name, nm/px it implies, relative difference) or
        (None, None, None) when no unit brings it within tolerance - which is
        a real disagreement, not an ambiguity about the glyph.
    """
    best = (None, None, None)
    for unit, factor in UNITS.items():
        implied = digits * factor / span
        off = abs(implied - stored) / stored
        if off <= TOLERANCE and (best[2] is None or off < best[2]):
            best = (unit, implied, off)
    return best


def main():
    import easyocr

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    reviewed = pd.read_csv(ANALYSIS / "reviewed_results.csv").set_index("file_name")
    review = pd.read_csv(ANALYSIS / "SEM_review.csv").set_index("image")

    paths = sorted(p for p in SOURCE.rglob("*.tif*")
                   if "reports" not in str(p) and "Reference" not in str(p))
    rows, tiles = [], []
    for path in paths:
        name = f"{path.parent.parent.name}_{path.parent.name}_{path.stem}"
        meta = metadata(path)
        image = np.array(Image.open(path).convert("L"))
        height, width = image.shape
        databar = height - meta["res_y"]
        strip = image[meta["res_y"]:, :]

        stored = float(reviewed.loc[f"{name}.tif", "nm_per_px"])
        trimmed = int(review.loc[f"{name}.tif", "databar_px"])
        from_pixel_width = meta["pixel_width_m"] * 1e9
        from_hfw = meta["hfw_m"] / meta["res_x"] * 1e9
        from_mag = REFERENCE_NM * REFERENCE_MAG / meta["magnification"]

        found = bar_span(strip)
        span = digits = None
        text, unit, from_bar, bar_off = "", None, None, None
        if found is not None:
            x0, x1, span = found
            digits, text = read_digits(reader, strip, x0, x1)
            if digits:
                unit, from_bar, bar_off = unit_that_fits(digits, span, stored)

        def off(value):
            return abs(value - stored) / stored

        checks = {
            "PixelWidth": off(from_pixel_width) <= 1e-4,
            "HFW/ResX": off(from_hfw) <= 1e-3,
            "magnification": off(from_mag) <= 1e-3,
            "printed bar": from_bar is not None,
            "square pixels": abs(meta["pixel_height_m"] - meta["pixel_width_m"]) < 1e-15,
            "databar trimmed": databar == trimmed,
            "bar is standard": from_bar is not None and any(
                abs(digits * UNITS[unit] - v) <= 0.02 * v for v in STANDARD),
        }
        rows.append({
            "frame": name, "mag": round(meta["magnification"]),
            "hv_kV": meta["hv_kv"], "stored_nm_per_px": stored,
            "PixelWidth": round(from_pixel_width, 4),
            "HFW_over_ResX": round(from_hfw, 4),
            "from_magnification": round(from_mag, 4),
            "bar_px": span, "ocr": text.strip()[:18], "digits": digits,
            "unit": unit,
            "bar_nm_per_px": None if from_bar is None else round(from_bar, 4),
            "bar_off_pct": None if bar_off is None else round(100 * bar_off, 3),
            "databar_px": databar, "trimmed_px": trimmed,
            "failed": ", ".join(k for k, ok in checks.items() if not ok),
        })

        if found is not None:
            pad = 14
            left = max(0, x0 - pad)
            crop = strip[:, left:min(width, x1 + pad)]
            scale = min(2.0, 940 / max(1, crop.shape[1]))
            tile = Image.fromarray(crop).convert("RGB").resize(
                (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
                Image.LANCZOS)
            draw = ImageDraw.Draw(tile)
            for x in (x0, x1):
                px = (x - left) * scale
                draw.line([(px, 0), (px, tile.height)], fill=(255, 60, 90), width=2)
            band = Image.new("RGB", (tile.width, 20), (10, 10, 10))
            colour = (120, 255, 140) if not rows[-1]["failed"] else (255, 120, 120)
            ImageDraw.Draw(band).text(
                (4, 4),
                f"{name}  {meta['magnification']:.0f}x  span {span}px  "
                f"stored {stored:.4f}  bar {rows[-1]['bar_nm_per_px']} nm/px "
                f"({rows[-1]['bar_off_pct']}%)",
                fill=colour)
            joined = Image.new("RGB", (tile.width, tile.height + 20))
            joined.paste(tile, (0, 0))
            joined.paste(band, (0, tile.height))
            tiles.append(joined)
        print(f"{name}: {meta['magnification']:.0f}x span {span} '{text.strip()[:14]}' "
              f"-> {rows[-1]['bar_nm_per_px']} vs {stored:.4f}"
              f"{'  FAILED ' + rows[-1]['failed'] if rows[-1]['failed'] else ''}",
              flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "sem_scale_audit.csv", index=False)
    table.to_csv(ANALYSIS / "sem_scale_audit.csv", index=False)

    per_sheet, columns = 14, 2
    for start in range(0, len(tiles), per_sheet):
        chunk = tiles[start:start + per_sheet]
        w = max(t.width for t in chunk)
        h = max(t.height for t in chunk)
        rows_n = (len(chunk) + columns - 1) // columns
        sheet = Image.new("RGB", (w * columns, h * rows_n), "white")
        for i, tile in enumerate(chunk):
            sheet.paste(tile, ((i % columns) * w, (i // columns) * h))
        sheet.save(OUT / f"sem_bars_{start // per_sheet + 1:02d}.png")

    print()
    print(table.groupby(["mag", "stored_nm_per_px"]).agg(
        frames=("frame", "size"),
        bar_px=("bar_px", lambda s: sorted(set(s.dropna()))),
        worst_bar_off_pct=("bar_off_pct", "max")).to_string())
    bad = table[table.failed != ""]
    print(f"\n{len(bad)} of {len(table)} frames failed a check")
    if len(bad):
        print(bad[["frame", "mag", "stored_nm_per_px", "bar_px", "ocr",
                   "bar_nm_per_px", "failed"]].to_string(index=False))


main()
