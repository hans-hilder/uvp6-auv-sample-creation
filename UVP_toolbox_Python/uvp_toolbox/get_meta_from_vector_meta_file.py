"""Match UVP sequences to vehicle metadata: position, yo number, sample names.

Translation of ``UVP6_samples_creation/GetMetaFromVectorMetaFile.m``.  The
MATLAB function contains two unrelated algorithms in one body:

* the original sequential matcher for SeaExplorer / SeaGlider / BGC-Argo
  floats, which walks the metadata files and the UVP sequences in lock-step
  (``get_meta_from_vector_meta_file``), and
* the ALR matcher added later, which walks every ALR segment file and picks
  the UVP sequence with the most images inside the segment's time span
  (``get_meta_alr``).

They are split into two functions here; the driver calls the right one.

Original MATLAB script by Camille Catalano, LOV, 2020/11
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .ctd_file import create_ctd_file
from .matlab_compat import finite_times, num2str
from .vector_io import (
    VectorMeta,
    convert_latlon_seaexplorer,
    read_data_seaexplorer,
    read_meta_alr,
    read_meta_float,
    read_meta_seaglider,
)

__all__ = [
    "VectorMatch",
    "AlrMatch",
    "get_meta_from_vector_meta_file",
    "get_meta_alr",
    "parse_alr_filename",
    "pick_best_sequence",
    "enforce_non_overlapping_indices",
]


@dataclass
class VectorMatch:
    """One entry per UVP sequence (legacy platforms)."""
    lon: np.ndarray
    lat: np.ndarray
    yo: np.ndarray
    sample_names: list[str]
    vector_filenames: list[str]


@dataclass
class AlrMatch:
    """One entry per matched ALR segment file."""
    lon: np.ndarray
    lat: np.ndarray
    yo: np.ndarray                      # ALR segment id (NaN when not in the filename)
    sample_names: list[str]             # ALR file basename
    vector_filenames: list[str]         # ALR file name with extension
    start_idx: np.ndarray               # 0-based first UVP image (as UVPapp expects)
    end_idx: np.ndarray                 # 0-based last UVP image
    start_time: np.ndarray              # datetime64[ns] of the first matched image
    assigned_uvp_seq_idx: np.ndarray    # 0-based index into the UVP sequence list
    alr_plot_time: np.ndarray           # concatenated ALR times (for diagnostics)
    alr_plot_depth: np.ndarray          # concatenated DEPTH_SCI
    skipped_count: int
    total_alr: int
    ctd_failures: list[tuple[str, str]] = field(default_factory=list)
    # Every ALR phase file, matched or not: name, letter, start, end, n_images (0 = no UVP image)
    phases: list[dict] = field(default_factory=list)


def _fmt04(x: float) -> str:
    """MATLAB ``num2str(x, '%04.f')``."""
    return "NaN" if math.isnan(x) else "%04.0f" % x


# ---------------------------------------------------------------------------
# Legacy sequential matcher (SeaExplorer / SeaGlider / float)
# ---------------------------------------------------------------------------

def _seaexplorer_yo(path: Path) -> float:
    m = re.search(r"\d+(?=\.gz)", path.name)
    return float(m.group(0)) if m else math.nan


def get_meta_from_vector_meta_file(
    vector_type: str,
    meta_data_folder,
    project_folder,
    ref_time_list,
    sequence_names: Sequence[str],
    profile_type_list: Sequence[str],
    cruise: str,
) -> VectorMatch:
    """Sequential matcher: assumes sequences and metadata files are both in
    chronological order and finds, for each sequence, the metadata file that
    covers its reference time (start of sequence, or end for floats).

    Writes one CTD file per matched sequence as a side effect, like MATLAB.
    """
    meta_data_folder = Path(meta_data_folder)
    reader: Callable[[Path], tuple[VectorMeta, object]]
    if vector_type == "SeaExplorer":
        files = list((meta_data_folder / "ccu" / "logs").glob("*raw*"))
        # ...8, 9, 10, 11... rather than ...1, 100, 101...; NaN sorts last as in MATLAB
        files.sort(key=lambda p: (math.isnan(_seaexplorer_yo(p)), _seaexplorer_yo(p)))
        reader = read_data_seaexplorer
    elif vector_type == "SeaGlider":
        files = sorted(meta_data_folder.glob("*.nc"))
        reader = read_meta_seaglider
    elif vector_type == "float":
        files = sorted(meta_data_folder.glob("S*.nc"))
        reader = read_meta_float
    elif vector_type == "ALR":
        raise ValueError("ALR projects use get_meta_alr()")
    else:
        raise ValueError(f"Unknown vector type {vector_type!r}")
    if not files:
        raise FileNotFoundError(f"No {vector_type} metadata files found under {meta_data_folder}")

    n = len(sequence_names)
    ref_time_list = np.asarray(ref_time_list, dtype="datetime64[ns]")
    ptypes = [str(p) for p in profile_type_list]
    lon = np.zeros(n)
    lat = np.zeros(n)
    yo = np.zeros(n)
    names = [""] * n
    vfn = [""] * n

    ten_s = np.timedelta64(10, "s")
    fifty_min = np.timedelta64(50, "m")
    thirty_min = np.timedelta64(30, "m")
    is_float = vector_type == "float"

    seq_nb = 0
    yo_nb = 0
    for path in files:
        meta, data = reader(path)
        if len(meta) == 0:
            raise ValueError(f"{path.name}: empty metadata file")
        t_first, t_last = meta.time[0], meta.time[-1]
        right_meta = True
        while right_meta and seq_nb < n:
            if ptypes[seq_nb] == "p":
                # a parking sequence takes the position of the next ascent
                later_ascents = [i for i in range(seq_nb + 1, n) if ptypes[i] == "a"]
                if not later_ascents:
                    raise ValueError(f"parking sequence {sequence_names[seq_nb]} has no following ascent")
                ref_seq_nb = later_ascents[0]
            else:
                ref_seq_nb = seq_nb
            time_to_find = ref_time_list[ref_seq_nb]
            names[seq_nb] = str(seq_nb + 1)

            in_file = bool((time_to_find + ten_s >= t_first) and (time_to_find <= t_last))
            in_file_float = bool(is_float and (time_to_find + fifty_min >= t_first)
                                 and (time_to_find - thirty_min <= t_last))
            if in_file or in_file_float:
                aa = np.flatnonzero(meta.time <= time_to_find)
                if aa.size == 0 and is_float:
                    aa = np.flatnonzero(meta.time <= time_to_find + fifty_min)
                elif aa.size == 0:
                    aa = np.flatnonzero(meta.time <= time_to_find + ten_s)
                print(f"Vector meta data for {sequence_names[seq_nb]} found")
                if aa.size == 0:
                    raise ValueError(f"{path.name}: no metadata record at or before {time_to_find}")
                k = aa[-1]
                ptype = ptypes[seq_nb]

                if vector_type == "SeaExplorer":
                    lat[seq_nb] = convert_latlon_seaexplorer(meta.lat[k])
                    lon[seq_nb] = convert_latlon_seaexplorer(meta.lon[k])
                    yo[seq_nb] = float(path.name.split(".")[-2])
                    names[seq_nb] = f"Yo_{_fmt04(yo[seq_nb])}{ptype}_{cruise}"

                elif vector_type == "SeaGlider":
                    lat[seq_nb] = meta.lat[k]
                    lon[seq_nb] = meta.lon[k]
                    if not (seq_nb > 0 and ptype == "a" and ptypes[seq_nb - 1] == "d"):
                        yo_nb += 1
                    yo[seq_nb] = yo_nb
                    # NB: MATLAB compares against the *unpadded* yo number here,
                    # so the '2' suffix only triggers for yo >= 1000.  Kept as is.
                    if seq_nb > 0 and names[seq_nb - 1] == f"Yo_{num2str(yo_nb)}{ptype}_{cruise}":
                        names[seq_nb] = f"Yo_{_fmt04(yo[seq_nb])}{ptype}2_{cruise}"
                    else:
                        names[seq_nb] = f"Yo_{_fmt04(yo[seq_nb])}{ptype}_{cruise}"

                elif vector_type == "float":
                    lat[seq_nb] = meta.lat[k]
                    lon[seq_nb] = meta.lon[k]
                    yo[seq_nb] = yo_nb
                    if ptype == "a":
                        yo_nb += 1
                    names[seq_nb] = f"{_fmt04(yo[seq_nb])}{ptype}_{cruise}"

                create_ctd_file(project_folder, data, names[seq_nb] + ".csv", vector_type)
                vfn[seq_nb] = path.name
                seq_nb += 1
            elif bool(time_to_find < t_first):
                seq_nb += 1
            else:
                right_meta = False
        if seq_nb >= n:
            break

    return VectorMatch(lon=lon, lat=lat, yo=yo, sample_names=names, vector_filenames=vfn)


# ---------------------------------------------------------------------------
# ALR matcher
# ---------------------------------------------------------------------------

_ALR_LETTER_RE = re.compile(r"_([A-Za-z])\d+\.nc$")


def parse_alr_filename(name: str) -> tuple[str, float, str]:
    """``ALR###_YYYYMMDD_uuuu_ssss_L####.nc`` -> ``(basename, segment_id, L)``.

    The segment id is the fourth run of digits in the basename (NaN if there
    are fewer than four) and the letter is the event type
    (``d``/``u``/``t``/``s``/``x``, or ``'na'`` when the suffix is missing).
    """
    base = Path(name).stem
    nums = re.findall(r"\d+", base)
    seg_id = float(nums[3]) if len(nums) >= 4 else math.nan
    m = _ALR_LETTER_RE.search(name)
    letter = m.group(1) if m else "na"
    return base, seg_id, letter


def pick_best_sequence(uvp_time_series: Sequence[np.ndarray], tmin, tmax) -> tuple[Optional[int], Optional[np.ndarray]]:
    """The UVP sequence with the most images inside ``[tmin, tmax]`` and the
    indices of those images; ``(None, None)`` when no sequence overlaps."""
    best_count = 0
    best_seq: Optional[int] = None
    best_idx: Optional[np.ndarray] = None
    for s, t in enumerate(uvp_time_series):
        if t.size == 0:
            continue
        idx = np.flatnonzero((t >= tmin) & (t <= tmax))
        if idx.size > best_count:
            best_count, best_seq, best_idx = idx.size, s, idx
    return best_seq, best_idx


def enforce_non_overlapping_indices(start_idx, end_idx, seq_idx) -> np.ndarray:
    """Trim each sample's ``end_idx`` so that, within one UVP sequence, image
    ranges never overlap the next sample's range (``end >= start`` is kept)."""
    start_idx = np.asarray(start_idx)
    end_idx = np.array(end_idx, copy=True)
    seq_idx = np.asarray(seq_idx)
    if start_idx.size == 0:
        return end_idx
    order = np.lexsort((start_idx, seq_idx))  # by sequence, then start; stable like sortrows
    prev_per_seq: dict[int, int] = {}
    for j in order:
        s = int(seq_idx[j])
        if s in prev_per_seq:
            p = prev_per_seq[s]
            e_new = min(end_idx[p], start_idx[j] - 1)
            e_new = max(e_new, start_idx[p])
            end_idx[p] = e_new
        prev_per_seq[s] = int(j)
    return end_idx


def get_meta_alr(
    meta_data_folder,
    project_folder,
    uvp_time_series: Sequence[np.ndarray],
    tolerance_seconds: float = 10.0,
    write_ctd: bool = True,
    verbose: bool = True,
) -> AlrMatch:
    """Walk every ALR segment ``.nc`` and match it to the UVP sequence with the
    most images within the segment's time span (+/- ``tolerance_seconds``).

    One CTD file per ALR segment is written as a side effect; failures are
    collected in ``AlrMatch.ctd_failures`` rather than swallowed.
    """
    files = sorted(Path(meta_data_folder).glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No ALR .nc files found in {meta_data_folder}")
    tol = np.timedelta64(int(round(tolerance_seconds * 1e9)), "ns")
    total_alr = len(files)

    lat_acc: list[float] = []
    lon_acc: list[float] = []
    yo_acc: list[float] = []
    name_acc: list[str] = []
    vfn_acc: list[str] = []
    sidx_acc: list[int] = []
    eidx_acc: list[int] = []
    stime_acc: list[np.datetime64] = []
    seq_acc: list[int] = []
    plot_time: list[np.ndarray] = []
    plot_depth: list[np.ndarray] = []
    ctd_failures: list[tuple[str, str]] = []
    phases: list[dict] = []           # every file, for the coverage check
    matched_phases: list[dict] = []   # the matched ones, in the order of the accumulators
    skipped_count = 0

    for k, path in enumerate(files, start=1):
        if verbose and k % 10 == 0:
            print(f"Processing ALR file {k}/{total_alr}: {path.name}")
        meta, data = read_meta_alr(path)
        t_alr = meta.time
        t_ok = finite_times(t_alr)
        phase = {"name": path.stem, "letter": parse_alr_filename(path.name)[2],
                 "start": t_ok.min() if t_ok.size else None, "end": t_ok.max() if t_ok.size else None,
                 "n_images": 0}
        phases.append(phase)
        if t_ok.size == 0:
            skipped_count += 1
            continue

        tmin = t_ok.min() - tol
        tmax = t_ok.max() + tol
        best_seq, best_idx = pick_best_sequence(uvp_time_series, tmin, tmax)
        if best_seq is None:
            skipped_count += 1
            continue
        matched_phases.append(phase)

        sidx = int(best_idx[0])
        eidx = int(best_idx[-1])
        stime = uvp_time_series[best_seq][sidx]

        lat_ok = np.flatnonzero(~np.isnan(meta.lat))
        lon_ok = np.flatnonzero(~np.isnan(meta.lon))
        lat_keep = float(meta.lat[lat_ok[0]]) if lat_ok.size else math.nan
        lon_keep = float(meta.lon[lon_ok[0]]) if lon_ok.size else math.nan

        base, seg_id, _ = parse_alr_filename(path.name)

        name_acc.append(base)
        vfn_acc.append(path.name)
        yo_acc.append(seg_id)
        lat_acc.append(lat_keep)
        lon_acc.append(lon_keep)
        sidx_acc.append(sidx)
        eidx_acc.append(eidx)
        stime_acc.append(stime)
        seq_acc.append(best_seq)

        if write_ctd:
            try:
                create_ctd_file(project_folder, data, base + ".csv", "ALR")
            except Exception as exc:  # MATLAB swallows this silently; we record it
                ctd_failures.append((path.name, f"{type(exc).__name__}: {exc}"))

        if "DEPTH_SCI" in data.columns:
            plot_time.append(np.asarray(t_alr, dtype="datetime64[ns]"))
            plot_depth.append(data["DEPTH_SCI"].to_numpy(dtype=float))

    end_idx = enforce_non_overlapping_indices(np.array(sidx_acc, dtype=int), np.array(eidx_acc, dtype=int), np.array(seq_acc, dtype=int))
    for phase, s, e in zip(matched_phases, sidx_acc, end_idx):
        phase["n_images"] = int(e) - int(s) + 1

    if not name_acc:
        warnings.warn(f"No ALR samples matched any UVP times within +/-{num2str(tolerance_seconds)} s.")

    return AlrMatch(
        lon=np.array(lon_acc, dtype=float),
        lat=np.array(lat_acc, dtype=float),
        yo=np.array(yo_acc, dtype=float),
        sample_names=name_acc,
        vector_filenames=vfn_acc,
        start_idx=np.array(sidx_acc, dtype=int),
        end_idx=end_idx,
        start_time=np.array(stime_acc, dtype="datetime64[ns]"),
        assigned_uvp_seq_idx=np.array(seq_acc, dtype=int),
        alr_plot_time=np.concatenate(plot_time) if plot_time else np.array([], dtype="datetime64[ns]"),
        alr_plot_depth=np.concatenate(plot_depth) if plot_depth else np.array([], dtype=float),
        skipped_count=skipped_count,
        total_alr=total_alr,
        ctd_failures=ctd_failures,
        phases=phases,
    )
