"""
One dataset out of six reviewed folders, named the way the samples are named.

Two naming systems met here. The SEM frames are filed by filter number - 02,
10, 12 - and the TEM frames by letter, one letter per filter with more than one
session on some of them. The letter is the sample; the number is the same
sample seen on the other instrument:

    A = filter 02     TEM sessions A1 (filed as O1) and A2
    B = filter 10     TEM sessions B1 and B2
    C = filter 12     TEM session C1

The folder on disk called O1 is A1. It is relabelled here rather than renamed
on disk, so the analysis folders still match the paths in every results file
that points at them.

Two tables come out, because the two answer different questions:

  particles   one row per measured object, for size distributions;
  frames      one row per reviewed frame including the empty ones, because a
              frame with nothing on it is an observation about the sample and
              dropping it would bias any per-frame count upwards.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, r"C:\Users\sanja\Projects\SAM-SEM-Segmentation\sem_particle_analysis")
from sem_particle_analysis.data_manager import parse_measurement_list

OUT = Path(sys.argv[1])

TEM = {
    "A1": ("A", "02", r"D:\NIOSH Sample Images\TEM Images\May20_2026 - TEM\O1_analysis"),
    "A2": ("A", "02", r"D:\NIOSH Sample Images\TEM Images\June3_2026 - TEM\A2_analysis"),
    "B1": ("B", "10", r"D:\NIOSH Sample Images\TEM Images\June3_2026 - TEM\B1_analysis"),
    "B2": ("B", "10", r"D:\NIOSH Sample Images\TEM Images\June11_2026 - TEM\B2_analysis"),
    "C1": ("C", "12", r"D:\NIOSH Sample Images\TEM Images\June11_2026 - TEM\C1_analysis"),
}
SEM = r"D:\NIOSH Sample Images\SEM Images\SEM_analysis"
# The SEM frame names carry the filter and where on the filter it was shot.
FILTER_TO_SAMPLE = {"02": "A", "10": "B", "12": "C"}


def rows_from(folder, label):
    """Every reviewed row of one folder, with its measurement lists parsed."""
    table = pd.read_csv(Path(folder) / "reviewed_results.csv")
    # On the stem, not the name: the same frame saved once as a TIFF and once
    # as the PNG copy the app reads is one frame, and comparing full names
    # missed exactly that in C1.
    stems = table.file_name.map(lambda n: Path(str(n)).stem)
    if stems.duplicated().any():
        raise SystemExit(f"{label}: {sorted(stems[stems.duplicated()])} appear "
                         f"more than once - refusing to combine")
    return table


def main():
    particles, frames = [], []

    for session, (sample, filter_id, folder) in TEM.items():
        for _, row in rows_from(folder, session).iterrows():
            frame = Path(str(row.file_name)).stem
            # The frames of session A1 are filed as O1_nnnn; the name is kept
            # so a row can still be traced to the file it came from.
            nm = float(row.nm_per_px)
            areas = parse_measurement_list(row.particle_areas_nm2)
            diameters = parse_measurement_list(row.equiv_diameters_nm)
            frames.append({"modality": "TEM", "sample": sample, "filter": filter_id,
                           "session": session, "position": None, "frame": frame,
                           "nm_per_px": nm, "n_particles": int(row.num_particles),
                           "scale_method": row.scale_method})
            for area, diameter in zip(areas, diameters):
                particles.append({"modality": "TEM", "sample": sample,
                                  "filter": filter_id, "session": session,
                                  "position": None, "frame": frame, "nm_per_px": nm,
                                  "area_nm2": area, "equiv_diameter_nm": diameter})

    for _, row in rows_from(SEM, "SEM").iterrows():
        frame = Path(str(row.file_name)).stem
        filter_id, position, _rest = frame.split("_", 2)
        sample = FILTER_TO_SAMPLE[filter_id]
        nm = float(row.nm_per_px)
        areas = parse_measurement_list(row.particle_areas_nm2)
        diameters = parse_measurement_list(row.equiv_diameters_nm)
        frames.append({"modality": "SEM", "sample": sample, "filter": filter_id,
                       "session": "SEM", "position": position, "frame": frame,
                       "nm_per_px": nm, "n_particles": int(row.num_particles),
                       "scale_method": row.scale_method})
        for area, diameter in zip(areas, diameters):
            particles.append({"modality": "SEM", "sample": sample, "filter": filter_id,
                              "session": "SEM", "position": position, "frame": frame,
                              "nm_per_px": nm, "area_nm2": area,
                              "equiv_diameter_nm": diameter})

    p = pd.DataFrame(particles)
    f = pd.DataFrame(frames)
    p["equiv_diameter_um"] = p.equiv_diameter_nm / 1000.0
    p["area_um2"] = p.area_nm2 / 1e6

    OUT.mkdir(parents=True, exist_ok=True)
    p.to_csv(OUT / "particles.csv", index=False)
    f.to_csv(OUT / "frames.csv", index=False)

    # Every frame's particle count must equal the rows it contributed.
    counted = p.groupby("frame").size()
    for _, row in f.iterrows():
        got = int(counted.get(row.frame, 0))
        assert got == row.n_particles, f"{row.frame}: {got} rows vs {row.n_particles}"

    print(f"{len(p)} particles across {len(f)} frames\n")
    print(f.groupby(["modality", "sample", "filter"]).agg(
        sessions=("session", "nunique"), frames=("frame", "size"),
        empty=("n_particles", lambda s: int((s == 0).sum())),
        particles=("n_particles", "sum")).to_string())
    print()
    print(f.groupby(["sample", "session"]).agg(
        frames=("frame", "size"), particles=("n_particles", "sum")).to_string())
    print()
    print(p.groupby(["modality", "sample"]).equiv_diameter_um.agg(
        n="size", median="median", mean="mean",
        p10=lambda s: s.quantile(0.10), p90=lambda s: s.quantile(0.90),
        smallest="min", largest="max").round(3).to_string())
    print(f"\nwrote {OUT/'particles.csv'} and {OUT/'frames.csv'}")


main()
