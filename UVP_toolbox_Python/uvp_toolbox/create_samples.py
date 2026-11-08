"""Create the UVPapp sample (header) file for every sequence of a project.

Translation of ``UVP6_samples_creation/uvp6_create_samples_alr.m`` — the
driver that reads each UVP6 sequence, finds the usable depth from the
instrument's black-image noise, matches sequences to the vehicle metadata
(SeaExplorer, SeaGlider, BGC-Argo float or ALR), writes one CTD file per
sample, writes ``meta/uvp6_header_<project>.txt`` and produces the diagnostic
figures.

Original MATLAB script by Camille Catalano, LOV, 2020/11
"""

from __future__ import annotations

import math
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .detection_vector_meta_file import detection_vector_meta_file
from .diagnostics import make_timedepth_plots, sampling_interval_diagnostics
from .qc import run_alr_qc
from .get_meta_from_vector_meta_file import get_meta_alr, get_meta_from_vector_meta_file, parse_alr_filename
from .matlab_compat import datestr_yyyymmdd_HHMMSS, num2str
from .usable_depth_limit import usable_depth_limit
from .uvp6_io import (
    datafile_to_array,
    read_data_from_dattable,
    read_metadata_from_hwline,
    read_metalines_from_datafile,
)
from .prompts import NullPrompter, Prompter

__all__ = [
    "VECTOR_TYPES",
    "SAMPLE_FILE_COLUMNS",
    "SequenceInfo",
    "SampleRow",
    "create_samples",
    "detect_vector_type",
    "sample_filename_for",
    "archive_existing_sample_file",
    "read_cruise_name",
    "list_sequences",
    "read_sequence",
    "write_sample_file",
]

VECTOR_TYPES = ("SeaExplorer", "SeaGlider", "float", "ALR")

SAMPLE_FILE_COLUMNS = [
    "cruise", "ship", "filename", "profileid",
    "bottomdepth", "ctdrosettefilename", "latitude", "longitude",
    "firstimage", "volimage", "aa", "exp",
    "dn", "winddir", "windspeed", "seastate",
    "nebuloussness", "comment", "endimg", "yoyo",
    "stationid", "sampletype", "integrationtime", "argoid",
    "pixelsize", "sampledatetime",
]

_RULE = "~" * 20


# ---------------------------------------------------------------------------
# Data carried per UVP sequence and per output row
# ---------------------------------------------------------------------------

@dataclass
class SequenceInfo:
    name: str
    folder: Path
    aa: float               # Aa / 1e6
    exp: float
    volimage: float
    pixelsize: float
    time: np.ndarray        # datetime64[ns] per image
    depth: np.ndarray       # m per image
    profile_type: str       # 'a' ascent, 'd' descent, 'p' parking
    sample_type: str        # 'P' depth-integrated, 'T' time-integrated
    integration_time: float
    start_idx: float        # 0-based first usable image (NaN if unknown)
    end_idx: float
    start_time: np.datetime64
    stop_time: np.datetime64
    zusable: float


@dataclass
class SampleRow:
    cruise: str
    ship: str
    filename: str
    profileid: str
    latitude: float
    longitude: float
    firstimage: float
    volimage: float
    aa: float
    exp: float
    endimg: float
    stationid: str
    sampletype: str
    integrationtime: float
    argoid: str
    pixelsize: float
    sampledatetime: np.datetime64

    def to_line(self) -> str:
        return ";".join([
            self.cruise,
            self.ship,
            self.filename,
            self.profileid,
            "nan",
            self.profileid + ".ctd",
            num2str(self.latitude),
            num2str(self.longitude),
            num2str(self.firstimage),
            num2str(self.volimage),
            num2str(self.aa),
            num2str(self.exp),
            "",
            "nan",
            "nan",
            "nan",
            "nan",
            "",
            num2str(self.endimg),
            "",
            self.stationid,
            self.sampletype,
            num2str(self.integrationtime),
            self.argoid,
            num2str(self.pixelsize),
            datestr_yyyymmdd_HHMMSS(self.sampledatetime),
        ])


# ---------------------------------------------------------------------------
# Project-level helpers
# ---------------------------------------------------------------------------

def detect_vector_type(project_folder, prompter: Optional[Prompter] = None) -> str:
    """Platform from the project folder *name* (MATLAB tests the whole path):
    ``sea`` -> SeaExplorer, ``SG`` -> SeaGlider, ``WMO`` -> float, or an
    ``alr_meta_sn*`` folder under ``doc`` -> ALR.  Otherwise asks ``prompter``
    (error when no prompter is given)."""
    project_folder = Path(project_folder)
    name = project_folder.name
    if "sea" in name:
        return "SeaExplorer"
    if "SG" in name:
        return "SeaGlider"
    if "WMO" in name:
        return "float"
    if any((project_folder / "doc").glob("alr_meta_sn*")):
        return "ALR"
    warnings.warn("Only Seaexplorer, Seaglider, Float, and ALR project types are supported")
    prompter = prompter or NullPrompter()
    answer = prompter.choose_option(
        "UVP project type",
        "Is it a Seaexplorer [se], Seaglider [sg], float [fl], " 
        "or ALR [alr] project?",
        ["se","sg","fl","alr"],"alr")
    if answer is None:
        raise RuntimeError("ERROR: Process has been aborted")
    mapping = {"": "ALR", "se": "SeaExplorer", "sg": "SeaGlider", "fl": "float", "alr": "ALR"}
    if answer not in mapping:
        raise ValueError("ERROR : The project is not a valid platform type")
    return mapping[answer]


def sample_filename_for(project_folder) -> str:
    """``uvp6_sn000227lp_...`` -> ``uvp6_header_sn000227lp_....txt``."""
    name = Path(project_folder).name
    return name[:5] + "header" + name[4:] + ".txt"


def archive_existing_sample_file(sample_path: Path, assume_yes: bool = False, now = None,
                                 prompter: Optional[Prompter] = None) -> Optional[Path]:
    """Rename an existing sample file with a timestamp suffix (after asking,
    unless ``assume_yes``).  Returns the archived path or ``None``."""
    if not sample_path.exists():
        return None
    warnings.warn("There is already a meta data file in \\meta. IT WILL BE ARCHIVED")
    if not assume_yes:
        prompter = prompter or NullPrompter()
        if not prompter.confirm(
            "Existing sample file",
            f"There is already a meta data file:\n\n{sample_path.name}\n\n"
            "It will be archived. Continue?", default=False):
            raise RuntimeError("ERROR: Process has been aborted")
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    archived = sample_path.with_name(f"{sample_path.stem}_{stamp}{sample_path.suffix}")
    sample_path.replace(archived)
    return archived


def read_cruise_name(project_folder) -> str:
    """Second line of ``config/cruise_info.txt`` from its 7th character on
    (``tline(7:end)``), whitespace-stripped; ``'unknown'`` when unreadable."""
    try:
        with open(Path(project_folder) / "config" / "cruise_info.txt", "r", encoding="latin-1") as f:
            lines = f.read().splitlines()
        return lines[1][6:].strip()
    except Exception:
        return "unknown"


def list_sequences(project_folder) -> list[Path]:
    """Sequence folders ``raw/20*`` in name order, excluding ``UsedForMerge``."""
    raw = Path(project_folder) / "raw"
    return sorted(p for p in raw.glob("20*") if p.is_dir() and "UsedForMerge" not in p.name)


# ---------------------------------------------------------------------------
# Per-sequence processing
# ---------------------------------------------------------------------------

def read_sequence(seq_dir: Path, vector_type: str, parking_pressure_diff: float,
                  deep_black_limit: float, verbose: bool = False) -> SequenceInfo:
    """Hardware metadata, per-image time/depth, profile direction and the
    usable-depth image range for one sequence."""
    seq_dat_file = seq_dir / f"{seq_dir.name}_data.txt"
    hw_line, _, _, _ = read_metalines_from_datafile(seq_dat_file)
    hw = read_metadata_from_hwline(hw_line)

    data, meta, _ = datafile_to_array(seq_dat_file)
    # Only black class 2 ("black 2pix") is needed for the noise detection.
    seq = read_data_from_dattable(meta, data, n_classes=2, want=("black",), verbose=verbose)
    time_data, depth_data = seq.time, seq.depth
    if seq.n_images == 0:
        raise ValueError(f"{seq_dat_file}: no image records")

    # black_nb = [depth time black]; drop rows that are not black images
    keep = ~np.isnan(seq.black[:, 0])
    black_depth = depth_data[keep]
    black_2pix = seq.black[keep, 1]

    integration_time = math.nan
    if vector_type == "float" and abs(depth_data[-1] - depth_data[0]) < parking_pressure_diff:
        profile_type, sample_type = "p", "T"
        integration_time = 1.0
    elif depth_data[-1] < depth_data[0]:
        profile_type, sample_type = "a", "P"
        black_depth = black_depth[::-1]
        black_2pix = black_2pix[::-1]
    else:
        profile_type, sample_type = "d", "P"

    zusable = usable_depth_limit(black_depth, black_2pix, deep_black_limit)

    if math.isnan(zusable):
        start_idx = end_idx = math.nan
        start_time, stop_time = time_data[0], time_data[-1]
    else:
        usable = np.flatnonzero(depth_data >= zusable)
        if usable.size == 0:
            raise ValueError(f"{seq_dir.name}: no image at or below the usable depth {zusable} m")
        start_idx, end_idx = int(usable[0]), int(usable[-1])  # already 0-based, as UVPapp expects
        start_time, stop_time = time_data[usable[0]], time_data[usable[-1]]

    return SequenceInfo(
        name=seq_dir.name, folder=seq_dir,
        aa=hw.aa / 1e6, exp=hw.exp, volimage=hw.volume, pixelsize=hw.pixel,
        time=time_data, depth=depth_data,
        profile_type=profile_type, sample_type=sample_type, integration_time=integration_time,
        start_idx=start_idx, end_idx=end_idx, start_time=start_time, stop_time=stop_time,
        zusable=zusable,
    )


# ---------------------------------------------------------------------------
# Sample file writing
# ---------------------------------------------------------------------------

def write_sample_file(path: Path, rows: list[SampleRow]) -> None:
    """``;``-separated, windows-1252, LF line endings — as MATLAB's
    ``fopen(..., 'w', 'n', 'windows-1252')`` + ``fprintf('%s\\n')``."""
    with open(path, "w", encoding="cp1252", errors="replace", newline="\n") as f:
        f.write(";".join(SAMPLE_FILE_COLUMNS) + "\n")
        for row in rows:
            f.write(row.to_line() + "\n")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def create_samples(
    project_folder,
    *,
    vector_type: Optional[str] = None,
    cruise: Optional[str] = None,
    parking_pressure_diff: float = 70.0,
    deep_black_limit: float = 100.0,
    plots_dir=None,
    make_plots: bool = True,
    qc_min_images: int = 20,
    assume_yes: bool = False,
    prompter: Optional[Prompter] = None,
    verbose: bool = False,
) -> Path:
    """Create the sample file for a UVP project and return its path.

    Parameters mirror the MATLAB script's inputs: ``parking_pressure_diff``
    (pressure difference identifying float parkings, 70 with margins) and
    ``deep_black_limit`` (depth below which the black level is instrumental,
    100 m).  ``cruise`` overrides ``config/cruise_info.txt``; ``plots_dir``
    is asked for with the ``prompter`` when not given.  ``qc_min_images`` is the
    sample size below which the quality checks report a sample as small (ALR only).
    """
    if project_folder is None:
        print("Select UVP project folder ")
        project_folder = (prompter or NullPrompter()).choose_directory("Select UVP project folder")
        if project_folder is None:
            raise RuntimeError("ERROR: No project folder selected")
    project_folder = Path(project_folder).expanduser().resolve()
    if not project_folder.is_dir():
        raise FileNotFoundError(f"Project folder not found: {project_folder}")
    if make_plots and plots_dir is None:
        plots_dir = (prompter or NullPrompter()).choose_directory(
            "Select output directory for figures", initial=project_folder)
        if plots_dir is None:
            raise RuntimeError("ERROR: No output folder is selected")

    print(_RULE)
    print("------------- UVP6 ALR sample creator --------------------")
    print(_RULE)
    print("WARNING : Python port of the 'UVP_Toolbox' https://github.com/ecotaxa/UVP_toolbox by ccatalano")
    print("Consult the README.md file for recommended project structure.")
    print(_RULE)
    print(f"Project folder : {project_folder}")
    if make_plots:
        print(f"Figures folder : {plots_dir}")
    print(_RULE)

    if vector_type is None:
        vector_type = detect_vector_type(project_folder, prompter=prompter)
    if vector_type not in VECTOR_TYPES:
        raise ValueError(f"vector_type must be one of {VECTOR_TYPES}, got {vector_type!r}")
    print(f"{vector_type} project")

    meta_data_folder, vector_sn = detection_vector_meta_file(project_folder, vector_type)

    meta_dir = project_folder / "meta"
    meta_dir.mkdir(exist_ok=True)
    sample_path = meta_dir / sample_filename_for(project_folder)
    archived = archive_existing_sample_file(sample_path, assume_yes=assume_yes, prompter=prompter)
    if archived is not None:
        print(f"Archived previous sample file as {archived.name}")
    print(_RULE)

    cruise_name = cruise if cruise is not None else read_cruise_name(project_folder)
    print(f"Cruise : {cruise_name}")

    seq_dirs = list_sequences(project_folder)
    if not seq_dirs:
        raise FileNotFoundError(f"No sequence folders found under {project_folder / 'raw'}")

    print("Get data from all sequences...")
    print(f"The instrumental noise is evaluated under {num2str(deep_black_limit)}m")
    seqs: list[SequenceInfo] = []
    for seq_dir in seq_dirs:
        seqs.append(read_sequence(seq_dir, vector_type, parking_pressure_diff, deep_black_limit, verbose=verbose))
        print(f"Sequence {seq_dir.name} done.")
    print(_RULE)

    print("Process the vector meta data....")
    uvp_time_series = [s.time for s in seqs]
    uvp_depth_series = [s.depth for s in seqs]
    ref_time_list = np.array([s.stop_time if vector_type == "float" else s.start_time for s in seqs], dtype="datetime64[ns]")

    rows: list[SampleRow] = []
    alr_time = alr_depth = None
    res = None
    qc_samples: list[tuple[str, str, int]] = []  # (sample, phase letter, images) for the quality checks
    if vector_type != "ALR":
        match = get_meta_from_vector_meta_file(
            vector_type, meta_data_folder, project_folder, ref_time_list,
            [s.name for s in seqs], [s.profile_type for s in seqs], cruise_name)
        for i, s in enumerate(seqs):
            rows.append(SampleRow(
                cruise=cruise_name, ship=vector_sn, filename=s.name, profileid=match.sample_names[i],
                latitude=float(match.lat[i]), longitude=float(match.lon[i]),
                firstimage=s.start_idx, volimage=s.volimage, aa=s.aa, exp=s.exp, endimg=s.end_idx,
                stationid=num2str(match.yo[i]), sampletype=s.sample_type, integrationtime=s.integration_time,
                argoid=match.vector_filenames[i], pixelsize=s.pixelsize, sampledatetime=s.start_time,
            ))
    else:
        res = get_meta_alr(meta_data_folder, project_folder, uvp_time_series, verbose=True)
        for i in range(len(res.sample_names)):
            s = seqs[int(res.assigned_uvp_seq_idx[i])]
            _, seg_id, letter = parse_alr_filename(res.vector_filenames[i])
            # Only 'u' and 'd' segments traverse the water column (depth
            # integration); anything else is time-integrated like a parking.
            if letter in ("u", "d"):
                sample_type, integration_time = "P", s.integration_time
            else:
                sample_type, integration_time = "T", 1.0
            station_id = "segment_NaN" if math.isnan(seg_id) else f"segment_{int(seg_id)}"
            qc_samples.append((res.sample_names[i], letter, int(res.end_idx[i]) - int(res.start_idx[i]) + 1))
            rows.append(SampleRow(
                cruise=cruise_name, ship=vector_sn, filename=s.name, profileid=res.sample_names[i],
                latitude=float(res.lat[i]), longitude=float(res.lon[i]),
                firstimage=int(res.start_idx[i]), volimage=s.volimage, aa=s.aa, exp=s.exp, endimg=int(res.end_idx[i]),
                stationid=station_id, sampletype=sample_type, integrationtime=integration_time,
                argoid=res.vector_filenames[i], pixelsize=s.pixelsize, sampledatetime=res.start_time[i],
            ))

        print(f"Matching UVP data times were not found for {res.skipped_count} of {res.total_alr} CTD/NAV files.")
        # Flag unmatched images
        matched = [np.zeros(s.time.size, dtype=bool) for s in seqs]
        for seq_i, first, last in zip(res.assigned_uvp_seq_idx, res.start_idx, res.end_idx):
            matched[seq_i][first:last + 1] = True
        n_images = sum(m.size for m in matched)
        n_unmatched = sum(int((~m).sum()) for m in matched)
        print(f"Matching CTD/NAV files were not found for {n_unmatched} of {n_images} UVP images")

        if res.ctd_failures:
            print(f"WARNING : {len(res.ctd_failures)} CTD file(s) could not be written:")
            for fname, err in res.ctd_failures[:10]:
                print(f"    {fname}: {err}")
        alr_time, alr_depth = res.alr_plot_time, res.alr_plot_depth

    print(_RULE)

    print("Creating the sample file...")
    write_sample_file(sample_path, rows)
    print(f"Sample file created : {sample_path}")
    print(_RULE)
    print("end of process")
    print(_RULE)

    if make_plots:
        outdir = Path(plots_dir)
        try:
            make_timedepth_plots(outdir, uvp_time_series, uvp_depth_series, alr_time, alr_depth)
        except Exception as exc:  # the MATLAB block is wrapped the same way
            print(f"Plotting skipped: {exc}")
        try:
            sampling_interval_diagnostics(outdir, uvp_time_series, alr_time)
        except Exception as exc:
            print(f"Sampling interval diagnostics skipped: {exc}")
        if res is not None:
            try:
                run_alr_qc(outdir, uvp_time_series, uvp_depth_series, res, qc_samples, min_images=qc_min_images)
            except Exception as exc:
                print(f"Quality checks skipped: {exc}")

    return sample_path
