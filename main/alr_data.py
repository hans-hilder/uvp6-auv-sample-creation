"""Input, building and output of ALR data.

Covers the whole data path, with no plotting:

    files  ->  parse/merge  ->  find_profiles  ->  annotate  ->  netCDF

Typical use:

    import alr_data

    ctd_files = alr_data.list_data_files("path/to/data", "*.log")
    nav_files = alr_data.list_data_files("path/to/data", "*.csv")

    big_df = alr_data.load_all_files(ctd_files, nav_files)
    df, profiles_df = alr_data.detect_profiles(big_df)
    big_df_with_profiles = alr_data.attach_profiles(df, profiles_df)
    alr_data.export_alr_netcdf(big_df_with_profiles, alr_serial="ALR005")

Plotting of these outputs lives in alr_plots.py.
"""

import io
import os
from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

import find_profiles_ALR as fpALR


# ---------------------------------------------------------------------------
# 2.1  Data directories
# ---------------------------------------------------------------------------

def list_data_files(directory, pattern):
    """List files matching `pattern` in `directory`, as resolved path strings.

    Note: order follows the filesystem's glob order, not sorted order, so that
    positional indexing (e.g. files[1]) matches the original notebook.
    """
    return [str(file.resolve()) for file in Path(directory).glob(pattern)]


# ---------------------------------------------------------------------------
# 2.2 / 2.3  Raw ALR CTD (.log) + science navigation (.csv) files
# ---------------------------------------------------------------------------

# Column layouts of the two sampling rates found inside a CTD .log file
_CTD_4HZ_COLUMNS = ["Timestamp", "Pressure"]
_CTD_1HZ_COLUMNS = ["Timestamp", "Conductivity", "Temperature", "Pressure", "Oxygen"]

_NAV_SCHEMA = {
    "timestamp": pl.Float64,
    "position_lat": pl.Float64,
    "position_lon": pl.Float64,
    "depth": pl.Float64,
    "roll": pl.Float64,
    "pitch": pl.Float64,
    "heading": pl.Float64,
    "altitude": pl.Float64,
}


def _segment_ctd_lines(lines):
    """Split CTD log lines into (segment_type, start, end) by '4 Hz'/'1 Hz' markers."""
    segments = []
    current_type = None
    current_start = None

    for i, line in enumerate(lines):
        if "4 Hz" in line:
            if current_type is not None:
                segments.append((current_type, current_start, i))
            current_type = "4hz"
            current_start = i + 1
        elif "1 Hz" in line:
            if current_type is not None:
                segments.append((current_type, current_start, i))
            current_type = "1hz"
            current_start = i + 1

    if current_type is not None:
        segments.append((current_type, current_start, len(lines)))

    return segments


def _read_ctd_segment(raw, seg_type):
    """Parse one raw CTD segment into a dataframe."""
    columns = _CTD_4HZ_COLUMNS if seg_type == "4hz" else _CTD_1HZ_COLUMNS
    return pl.read_csv(
        io.StringIO(raw),
        has_header=False,
        separator=",",
        new_columns=columns,
        schema_overrides={c: pl.Float64 for c in columns},
        null_values=["", "NaN", "S>"],
        ignore_errors=True,
    )


def _merge_ctd_nav(all_data, nav_data):
    """Join CTD and nav data on Timestamp and interpolate Pressure over the gaps."""
    all_data = all_data.with_columns(pl.col("Timestamp").cast(pl.Float64))
    nav_data = nav_data.rename({"timestamp": "Timestamp"}).with_columns(
        pl.col("Timestamp").cast(pl.Float64)
    )

    combined = all_data.join(nav_data, on="Timestamp", how="full")  # union of both

    # Collapse Timestamp + Timestamp_right into a single column
    if "Timestamp_right" in combined.columns:
        combined = combined.with_columns(
            pl.coalesce([pl.col("Timestamp"), pl.col("Timestamp_right")]).alias("Timestamp")
        ).drop("Timestamp_right")

    combined = combined.sort("Timestamp")

    # Interpolate depth/pressure values over time gaps
    combined = combined.with_columns(pl.col("Pressure").interpolate_by("Timestamp"))

    return combined


def load_file_pair(ctd_file, nav_file):
    """Parse one CTD .log and one nav .csv, and combine them into one dataframe.

    This is section 2.2: a single file set, useful for inspecting one deployment.
    """
    with open(ctd_file, "r") as f:
        lines = f.readlines()

    segments = _segment_ctd_lines(lines)

    parsed_segments = []
    for seg_type, start, end in segments:
        raw = "".join(lines[start:end])
        parsed_segments.append(_read_ctd_segment(raw, seg_type))

    all_data = pl.concat(parsed_segments, how="diagonal").sort("Timestamp")
    nav_data = pl.read_csv(nav_file, schema_overrides=_NAV_SCHEMA)

    return _merge_ctd_nav(all_data, nav_data)


def load_all_files(ctd_files, nav_files, verbose=False):
    """Parse every CTD and nav file and merge them into a single dataframe.

    This is section 2.3. Files that fail to open, have no 4 Hz / 1 Hz markers,
    or carry the wrong column count are reported and skipped rather than
    aborting the run. Set verbose=True to see that per-file commentary.

    Returns a dataframe with a 'row_id' index column.
    """
    def log(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    # --- Collect all CTD data ---
    ctd_runs = []

    for ctd_file in ctd_files:
        log(f"[CTD] Reading file: {ctd_file}")
        try:
            with open(ctd_file, "r") as f:
                lines = f.readlines()
        except Exception as e:
            log(f"[CTD] ERROR opening {ctd_file}: {e} — skipping")
            continue

        segments = _segment_ctd_lines(lines)

        # Pre-clean: within 1hz segments, blank any line without exactly 5 fields
        for seg_type_tmp, start_tmp, end_tmp in segments:
            if seg_type_tmp == "1hz":
                for j in range(start_tmp, end_tmp):
                    ln = lines[j].strip()
                    if not ln:
                        continue
                    if ln.count(",") != 4 or len(ln.split(",")) != 5:
                        log(f"[CTD] Skipping non-5col line in 1hz at file_line={j}: {ln!r}")
                        lines[j] = ""

        log(f"[CTD] Detected segments ({len(segments)}): {segments}")
        if not segments:
            log(f"[CTD] WARNING {ctd_file}: no 4 Hz or 1 Hz markers found — skipping")
            continue

        # Validate column counts for each segment before parsing
        invalid_file = False
        for seg_idx, (seg_type, start, end) in enumerate(segments):
            expected_cols = 2 if seg_type == "4hz" else 5
            seg_lines = [ln.strip() for ln in lines[start:end] if ln.strip() != ""]
            preview = seg_lines[:5]
            log(
                f"[CTD] Validating seg#{seg_idx} type={seg_type} "
                f"range=({start},{end}) lines={len(seg_lines)} expected_cols={expected_cols}"
            )
            for pi, pln in enumerate(preview):
                log(f"        preview[{pi}] cols={len(pln.split(','))}: {pln!r}")

            sampled = seg_lines[:50] if len(seg_lines) > 50 else seg_lines
            mismatches = [ln for ln in sampled if len(ln.split(",")) != expected_cols]

            if len(seg_lines) == 0:
                log(f"[CTD] WARNING {ctd_file}: empty {seg_type} segment — skipping file")
                invalid_file = True
                break
            if mismatches:
                log(
                    f"[CTD] WARNING {ctd_file}: seg#{seg_idx} ({seg_type}) incorrect "
                    f"column count (expected {expected_cols}). Showing first 5 mismatches:"
                )
                for mi, bad in enumerate(mismatches[:5]):
                    log(f"        mismatch[{mi}] cols={len(bad.split(','))}: {bad!r}")
                invalid_file = True
                break

        if invalid_file:
            continue

        parsed_segments = []
        for seg_idx, (seg_type, start, end) in enumerate(segments):
            raw = "".join(lines[start:end])
            if seg_type == "1hz":
                # Rebuild raw ensuring only valid 5-col rows remain
                raw = "".join([ln for ln in lines[start:end] if ln.strip()])

            raw_preview = "\n".join([ln for ln in raw.splitlines()[:5]])
            log(f"[CTD] Parsing seg#{seg_idx} type={seg_type} chars={len(raw)} preview:\n{raw_preview}")

            try:
                df = _read_ctd_segment(raw, seg_type)
                log(f"[CTD] seg#{seg_idx} parsed OK: rows={df.height} cols={df.width} -> {df.columns}")
            except Exception as e:
                log(f"[CTD] ERROR parsing {ctd_file} seg#{seg_idx} ({seg_type}): {e} — skipping file")
                parsed_segments = []
                break

            parsed_segments.append(df)

        if len(parsed_segments) == 0:
            continue

        run_df = pl.concat(parsed_segments, how="diagonal").sort("Timestamp")
        run_df = run_df.with_columns(pl.lit(ctd_file).alias("source_ctd"))
        log(f"[CTD] OK: {ctd_file} parsed with {run_df.height} rows")
        ctd_runs.append(run_df)

    all_ctd = pl.concat(ctd_runs, how="diagonal").sort("Timestamp")

    # --- Collect all NAV data ---
    nav_runs = []
    for nav_file in nav_files:
        log(f"[NAV] Reading file: {nav_file}")
        try:
            df = pl.read_csv(nav_file, schema_overrides=_NAV_SCHEMA)
        except Exception as e:
            log(f"[NAV] ERROR opening/parsing {nav_file}: {e} — skipping")
            continue

        log(f"[NAV] Columns: {list(df.columns)} rows={df.height}")
        missing = set(_NAV_SCHEMA).difference(set(df.columns))
        if missing:
            log(f"[NAV] WARNING {nav_file}: missing required columns: {sorted(missing)} — skipping")
            continue

        df = df.rename({"timestamp": "Timestamp"})
        df = df.with_columns(
            pl.col("Timestamp").cast(pl.Float64),
            pl.lit(nav_file).alias("source_nav"),
        )
        log(f"[NAV] OK: {nav_file} parsed with {df.height} rows")
        nav_runs.append(df)

    all_nav = pl.concat(nav_runs, how="diagonal").sort("Timestamp")

    # --- Merge CTD + NAV across all files ---
    big_df = all_ctd.join(all_nav, on="Timestamp", how="full")

    if "Timestamp_right" in big_df.columns:
        big_df = big_df.with_columns(
            pl.coalesce([pl.col("Timestamp"), pl.col("Timestamp_right")]).alias("Timestamp")
        ).drop("Timestamp_right")

    big_df = big_df.sort("Timestamp")

    if big_df["Pressure"].drop_nulls().len() > 0:
        big_df = big_df.with_columns(pl.col("Pressure").interpolate_by("Timestamp"))

    # QC: drop rows where BOTH Pressure and pitch are null
    big_df = big_df.filter(~(pl.col("Pressure").is_null() & pl.col("pitch").is_null()))

    return big_df.with_row_index(name="row_id")


# ---------------------------------------------------------------------------
# 2.3.1  Already formatted data files (e.g. TechOceans deployment)
# ---------------------------------------------------------------------------

_PREFORMATTED_SCHEMA = {
    "Date": pl.Utf8,
    "Latitudedegrees_north": pl.Float64,
    "Longitudedegrees_east": pl.Float64,
    "Depthm": pl.Float64,
    "roll": pl.Float64,
    "pitch": pl.Float64,
    "heading": pl.Float64,
    "ConductivitymScm": pl.Float64,
    "TemperatureC": pl.Float64,
    "pbar": pl.Float64,
    "oxygen_1Hz": pl.Float64,
    "SP": pl.Float64,
    "Oxygen [mL/L]": pl.Float64,
    "Oxsol (T,S)": pl.Float64,
    "Oxygen (umol/L)": pl.Float64,
}

# Map the pre-formatted column names onto the names the rest of the pipeline uses
_PREFORMATTED_RENAMES = {
    "Latitudedegrees_north": "position_lat",
    "Longitudedegrees_east": "position_lon",
    "Depthm": "depth",
    "ConductivitymScm": "Conductivity",
    "TemperatureC": "Temperature",
    "pbar": "Pressure",
    "oxygen_1Hz": "Oxygen",
}


def load_preformatted_csv(path, verbose=False):
    """Load an already-formatted CSV (e.g. ODV export) into the pipeline's schema.

    Produces the same shape of dataframe as load_all_files().
    """
    big_df = pl.read_csv(
        path,
        separator=",",
        schema_overrides=_PREFORMATTED_SCHEMA,
        null_values=["", "NaN"],
    )

    big_df = big_df.rename(_PREFORMATTED_RENAMES)

    # ISO-8601 timestamp -> Unix epoch seconds (Float64)
    big_df = big_df.with_columns(
        pl.col("Date")
        .str.to_datetime(format="%Y-%m-%dT%H:%M:%SZ", time_unit="us")
        .dt.epoch(time_unit="s")
        .cast(pl.Float64)
        .alias("Timestamp")
    ).drop("Date")

    # altitude is not present in this file; add null column for schema parity
    big_df = big_df.with_columns(pl.lit(None).cast(pl.Float64).alias("altitude"))

    # Interpolate Pressure over any gaps (mirrors raw pipeline behaviour)
    if big_df["Pressure"].drop_nulls().len() > 0:
        big_df = big_df.with_columns(pl.col("Pressure").interpolate_by("Timestamp"))

    # QC: drop rows where BOTH Pressure and pitch are null
    big_df = big_df.filter(~(pl.col("Pressure").is_null() & pl.col("pitch").is_null()))

    big_df = big_df.with_row_index(name="row_id")

    if verbose:
        print(f"big_df ready: {big_df.height} rows, {big_df.width} columns")
        print(big_df.columns)

    return big_df


# ---------------------------------------------------------------------------
# 3.1  Profile analysis (wraps find_profiles_ALR)
# ---------------------------------------------------------------------------

def detect_profiles(
    big_df,
    gradient_thresholds=[0.05, -0.05],
    cust_gradient_thresholds=[0.005, -0.005],
    custom_column="pitch",
    filter_win_sizes=["30s", "20s"],
    transect_duration="5m",
    transect_depth_range=[10, 20, 30],
    transect_depth_bottom_limits=[20, 50],
    profile_duration="1m",
    strict_profiles=False,
    use_only_pit_vel=False,
    back_fill_mod=1,
    backfill_segments=True,
    force_split_up_down_casts=False,
):
    """Run find_profiles_ALR on a built dataframe.

    Renames Timestamp -> TIME and Pressure -> DEPTH (pressure is used as the
    profiling variable), converts TIME to Datetime[ns], then runs the algorithm.

    Returns (df, profiles_df):
        df          - the renamed input, needed later by attach_profiles()
        profiles_df - the algorithm output, one row per input row
    """
    # Rename columns to what find_profiles expects
    df = big_df.rename({
        "Timestamp": "TIME",
        "Pressure": "DEPTH",   # treat pressure as depth for profiling
    })

    # Convert TIME (epoch seconds float) -> Datetime[ns]
    df = df.with_columns((pl.col("TIME") * 1e9).cast(pl.Datetime("ns")).alias("TIME"))

    profiles_df = fpALR.find_profiles(
        df,
        gradient_thresholds=gradient_thresholds,
        filter_win_sizes=filter_win_sizes,
        time_col="TIME",
        depth_col="DEPTH",
        cust_col=custom_column,
        cust_gradient_thresholds=cust_gradient_thresholds,
        transect_duration=transect_duration,
        transect_depth_range=transect_depth_range,
        transect_depth_bottom_limits=transect_depth_bottom_limits,
        profile_duration=profile_duration,
        strict_profiles=strict_profiles,
        use_only_pit_vel=use_only_pit_vel,
        back_fill_mod=back_fill_mod,
        backfill_segments=backfill_segments,
        force_split_up_down_casts=force_split_up_down_casts,
    )

    profiles_df = profiles_df.drop_nans("is_profile")

    return df, profiles_df


def attach_profiles(df, profiles_df):
    """Join the profile/transect annotations back onto the full dataframe."""
    # Ensure both sides have matching TIME column format
    df = df.with_columns(pl.col("TIME").cast(pl.Datetime("ns")))
    profile_annotations = profiles_df.with_columns(pl.col("TIME").cast(pl.Datetime("ns")))

    big_df_with_profiles = df.join(profile_annotations, on=["TIME", "DEPTH"], how="left")

    return big_df_with_profiles.filter(pl.col("TIME").is_not_null())


# ---------------------------------------------------------------------------
# 4.0  Export to netCDF
# ---------------------------------------------------------------------------
#
# Filenames include the YYYYMMDD (UTC) of the event's first datapoint:
#     ALR004_20240401_0082_0029_d0020.nc
#
# All data are exported: alongside profiles (d/u) and transects (t/s), one 'x'
# event per segment is created for rows that are neither. Segment '0000' is
# everything prior to the first surface-origin downcast; subsequent segments
# increment at each such downcast start.


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _first_last_depth_for_profile(g: pl.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    g_sorted = g.sort("TIME")
    if "DEPTH" in g.columns:
        d = g_sorted.get_column("DEPTH").drop_nulls()
        if d.len() > 0:
            return float(d[0]), float(d[-1])
    if "depth" in g.columns:
        d = g_sorted.get_column("depth").drop_nulls()
        if d.len() > 0:
            return float(d[0]), float(d[-1])
    return None, None


def _mean_depth(g: pl.DataFrame) -> Optional[float]:
    if "DEPTH" in g.columns:
        m = g.select(pl.col("DEPTH")).mean().item()
        if m is not None:
            return float(m)
    if "depth" in g.columns:
        m = g.select(pl.col("depth")).mean().item()
        if m is not None:
            return float(m)
    return None


def _event_type_for_profile(g: pl.DataFrame) -> str:
    first_d, last_d = _first_last_depth_for_profile(g)
    if first_d is None or last_d is None:
        return "d"
    if first_d < last_d:
        return "d"
    elif first_d > last_d:
        return "u"
    else:
        return "d"


def _event_type_for_transect(g: pl.DataFrame, surface_thresh: float) -> str:
    m = _mean_depth(g)
    if m is None:
        return "t"
    return "s" if m < surface_thresh else "t"


def _dataset_from_group(g: pl.DataFrame) -> xr.Dataset:
    rename_map = {
        "DEPTH": "DEPTH_SCI",
        "depth": "DEPTH_ENG",
        "INTERP_DEPTH": "DEPTH_SCI_INTERP",
        "position_lat": "LAT",
        "position_lon": "LON",
        "TIME": "TIME",
        "Temperature": "TEMPERATURE",
        "Conductivity": "CONDUCTIVITY",
        "Oxygen": "OXYGEN",
        "pitch": "PITCH_ENG",
        "roll": "ROLL_ENG",
    }
    keep_cols = [c for c in rename_map.keys() if c in g.columns]
    if "TIME" not in keep_cols:
        raise ValueError("Required column 'TIME' missing from group.")

    g2 = (
        g.sort("TIME")
         .select(keep_cols)
         .rename({k: v for k, v in rename_map.items() if k in keep_cols})
    )

    pdf = g2.to_pandas()
    pdf["TIME"] = pd.to_datetime(pdf["TIME"], utc=True, errors="coerce")
    pdf = pdf.dropna(subset=["TIME"])

    epoch = np.datetime64("1970-01-01T00:00:00")
    time_seconds = (pdf["TIME"].values.astype("datetime64[ns]") - epoch) / np.timedelta64(1, "s")

    ds = xr.Dataset(
        data_vars={
            # the renamed columns (#15): TEMPERATURE, CONDUCTIVITY, OXYGEN, PITCH_ENG, ROLL_ENG were dropped before
            k: (("TIME",), pdf[k].to_numpy())
            for k in rename_map.values()
            if k != "TIME" and k in pdf.columns
        },
        coords={"TIME": time_seconds},
    )

    ds["TIME"].attrs.update({
        "units": "seconds since 1970-01-01 00:00:00 UTC",
        "long_name": "time",
        "standard_name": "time",
    })

    ds.attrs.update({
        "Conventions": "CF-1.9",
        "title": "ALR profile/transect/other export",
        "institution": "NOCS",
        "source": "Polars→xarray export",
        "history": "Created by export_alr_netcdf()",
        "author": "hans.hilder@noc.ac.uk",
    })
    return ds


def export_alr_netcdf(
    df: pl.DataFrame,
    *,
    alr_serial: str = "ALR000",
    surface_thresh: float = 5.0,
    output_dir: str = "output_ALR000_netCDF",
    max_files: Optional[int] = 5,
) -> List[str]:
    """
    Exports per-profile (d/u), per-transect (t/s), and per-'other' contiguous run (x) NetCDF files.

    Filename:
        {alr_serial}_{YYYYMMDD}_{unique_id:04d}_{segment_id:04d}_{etype}{etype_idx:04d}.nc

    Segment ID:
        0000 for anything before the first downcast that begins shallower than `surface_thresh`.
        Increments by 1 at each such downcast start thereafter.
    """
    _ensure_dir(output_dir)

    must_have = ["TIME", "position_lat", "position_lon"]
    missing_req = [c for c in must_have if c not in df.columns]
    if missing_req:
        raise ValueError(f"Missing required columns: {missing_req}")

    maybe_cols = [
        "DEPTH", "depth", "INTERP_DEPTH",
        "profile_num_new", "transect_num",
        "is_profile", "is_transect",
        "Temperature", "Conductivity", "Oxygen",
        "pitch", "roll",
    ]
    subset_cols = list(dict.fromkeys(must_have + [c for c in maybe_cols if c in df.columns]))

    # Sort and index early so we can split 'x' by contiguous runs.
    sdf = (
        df.select(subset_cols)
          .with_columns(pl.col("TIME").cast(pl.Datetime).alias("TIME"))
          .sort("TIME")
          .with_row_index("orig_idx")
    )

    # ---------- Build profile groups & collect downcast starts ----------
    profile_groups = []
    downcast_start_times_utc: List[pd.Timestamp] = []

    if "is_profile" in sdf.columns and "profile_num_new" in sdf.columns:
        prof_df = sdf.filter(pl.col("is_profile") == 1).drop_nulls(["profile_num_new"])
        if prof_df.height > 0:
            for key, g in prof_df.group_by("profile_num_new", maintain_order=False):
                start_time = pd.to_datetime(g.select(pl.col("TIME").min()).item(), utc=True)
                etype = _event_type_for_profile(g)
                profile_groups.append({
                    "kind": "profile",
                    "key": int(key[0]) if isinstance(key, tuple) else int(key),
                    "start_time": start_time,
                    "etype": etype,
                    "group": g.drop("orig_idx"),
                })
                if etype == "d":
                    first_d, _ = _first_last_depth_for_profile(g)
                    if first_d is not None and first_d < surface_thresh:
                        downcast_start_times_utc.append(start_time)

    # Keep these sorted for bisect-based segment lookup
    downcast_start_times_utc.sort()

    # ---------- Build transect groups ----------
    transect_groups = []
    if "is_transect" in sdf.columns and "transect_num" in sdf.columns:
        tran_df = sdf.filter(pl.col("is_transect") == 1).drop_nulls(["transect_num"])
        if tran_df.height > 0:
            for key, g in tran_df.group_by("transect_num", maintain_order=False):
                start_time = pd.to_datetime(g.select(pl.col("TIME").min()).item(), utc=True)
                etype = _event_type_for_transect(g, surface_thresh=surface_thresh)
                transect_groups.append({
                    "kind": "transect",
                    "key": int(key[0]) if isinstance(key, tuple) else int(key),
                    "start_time": start_time,
                    "etype": etype,
                    "group": g.drop("orig_idx"),
                })

    # ---------- Compute per-row segment_id (join_asof on downcast starts) ----------
    if downcast_start_times_utc:
        # Make naive-UTC datetimes for Polars join
        seg_starts = [pd.Timestamp(t).tz_convert("UTC").tz_localize(None).to_pydatetime()
                      for t in downcast_start_times_utc]
        seg_df = pl.DataFrame({"SEG_START": seg_starts}).with_row_index("segment_id", offset=1)
        sdf_with_seg = sdf.join_asof(
            seg_df.sort("SEG_START"),
            left_on="TIME", right_on="SEG_START",
            strategy="backward"
        ).with_columns(pl.col("segment_id").fill_null(0))
    else:
        sdf_with_seg = sdf.with_columns(pl.lit(0).alias("segment_id"))

    # ---------- Build 'x' groups: contiguous runs within each segment ----------
    # Rows that are neither profile nor transect
    not_flagged = (
        (pl.col("is_profile") != 1) & (pl.col("is_transect") != 1)
        if "is_profile" in sdf_with_seg.columns and "is_transect" in sdf_with_seg.columns
        else pl.lit(True)
    )

    other_df = (
        sdf_with_seg
        .filter(not_flagged)
        .select([c for c in ["TIME", "position_lat", "position_lon", "DEPTH", "depth",
                             "INTERP_DEPTH", "segment_id", "orig_idx", "Temperature",
                             "Conductivity", "Oxygen", "pitch", "roll"]
                 if c in sdf_with_seg.columns])
        .sort(["segment_id", "orig_idx"])
        .with_columns([
            # new run when segment changes OR original index is not consecutive
            (
                (pl.col("segment_id").diff().fill_null(0) != 0)
                | (pl.col("orig_idx").diff() != 1)
            ).fill_null(True).cast(pl.Int8).alias("new_run")
        ])
        .with_columns([
            pl.col("new_run").cum_sum().alias("x_run_id")
        ])
    )

    x_groups = []
    if other_df.height > 0:
        for (seg_id, run_id), g in other_df.group_by(["segment_id", "x_run_id"], maintain_order=True):
            start_time = pd.to_datetime(g.select(pl.col("TIME").min()).item(), utc=True)
            x_groups.append({
                "kind": "other",
                "key": (int(seg_id), int(run_id)),
                "start_time": start_time,
                "etype": "x",
                "group": g.drop(["segment_id", "orig_idx", "new_run", "x_run_id"]),
                "segment_id_override": int(seg_id),
            })

    # ---------- Merge & sort all events ----------
    events = profile_groups + transect_groups + x_groups
    if not events:
        return []

    events.sort(key=lambda e: e["start_time"])

    # ---------- Write files ----------
    unique_id = 0
    per_type_counts: Dict[str, int] = {"d": 0, "u": 0, "t": 0, "s": 0, "x": 0}
    written: List[str] = []

    for ev in events:
        if max_files is not None and len(written) >= max_files:
            break

        etype = ev["etype"]

        # Segment ID lookup: count how many downcast starts <= this event's start_time
        seg_id_for_file = ev.get(
            "segment_id_override",
            bisect_right(downcast_start_times_utc, ev["start_time"])
        )

        per_type_counts[etype] += 1
        unique_id += 1

        ds = _dataset_from_group(ev["group"])

        date_str = ev["start_time"].strftime("%Y%m%d")
        fname = f"{alr_serial}_{date_str}_{unique_id:04d}_{seg_id_for_file:04d}_{etype}{per_type_counts[etype]:04d}.nc"
        fpath = os.path.join(output_dir, fname)

        ds.attrs.update({
            "alr_serial": alr_serial,
            "unique_id": unique_id,
            "segment_id": seg_id_for_file,
            "event_type": etype,         # d/u/t/s/x
            "group_kind": ev["kind"],    # profile/transect/other
            "group_key": ev["key"],
            "surface_threshold_m": surface_thresh,
            "event_start_utc": ev["start_time"].isoformat(),
        })

        ds.to_netcdf(fpath, engine="netcdf4", format="NETCDF4")
        written.append(fpath)

    return written
