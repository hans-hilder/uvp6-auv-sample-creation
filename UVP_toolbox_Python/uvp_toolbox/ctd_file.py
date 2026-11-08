"""CTD file creation for UVPapp / EcoPart.

Translation of ``UVP6_samples_creation/CreateCTDfile.m`` and the four
``FillCTDtable*.m`` platform fillers.  A CTD file is a tab-separated table
with the 18 UVPapp column names, written in Latin-1, followed by whatever
extra columns the platform reader produced.

Original MATLAB script by Camille Catalano, LOV, 2020/11
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .matlab_compat import datestr_yyyymmddHHMMSSFFF, epoch_seconds_to_datetime64
from .seawater import sw_dens

__all__ = [
    "CTD_COLUMN_NAMES",
    "create_ctd_file",
    "write_table_tab",
    "fill_ctd_table_alr",
    "fill_ctd_table_float",
    "fill_ctd_table_seaexplorer",
    "fill_ctd_table_seaglider",
]

CTD_COLUMN_NAMES = [
    "chloro fluo [mg chl m-3]", "conductivity [ms cm-1]",
    "cpar [%]", "depth [m]", "fcdom [ppb qse]",
    "in situ density anomaly [kg m-3]", "nitrate [umol l-1]",
    "oxygen [umol kg-1]", "oxygen [ml l-1]", "par [umol m-2 s-1]",
    "potential density anomaly [kg m-3]", "potential temperature [degc]",
    "practical salinity [psu]", "pressure [db]", "qc flag",
    "spar [umol m-2 s-1]", "temperature [degc]", "time [yyyymmddhhmmssmmm]",
]

TIME_COLUMN = "time [yyyymmddhhmmssmmm]"


# ---------------------------------------------------------------------------
# writetable(..., 'Delimiter', 'tab') in Latin-1
# ---------------------------------------------------------------------------

def _format_column(s: pd.Series) -> list[str]:
    """Text for one column the way MATLAB ``writetable`` would print it:
    ``%.15g`` for doubles, ``%.7g`` for singles, ``NaN`` for missing."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return ["" if pd.isna(v) else pd.Timestamp(v).isoformat() for v in s]
    if pd.api.types.is_bool_dtype(s):
        return ["1" if v else "0" for v in s]
    if pd.api.types.is_integer_dtype(s):
        return [str(int(v)) for v in s]
    if pd.api.types.is_float_dtype(s):
        fmt = "%.7g" if s.dtype == np.float32 else "%.15g"
        return ["NaN" if math.isnan(v) else fmt % v for v in s.to_numpy(dtype=float)]
    out = []
    for v in s:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            out.append("NaN")
        else:
            out.append(str(v))
    return out


def write_table_tab(df: pd.DataFrame, path, encoding: str = "latin-1") -> None:
    """Write ``df`` as a tab-separated text table with LF line endings."""
    columns = [_format_column(df[c]) for c in df.columns]
    with open(path, "w", encoding=encoding, errors="replace", newline="\n") as f:
        f.write("\t".join(str(c) for c in df.columns) + "\n")
        for row in zip(*columns):
            f.write("\t".join(row) + "\n")


# ---------------------------------------------------------------------------
# FillCTDtable*
# ---------------------------------------------------------------------------

def fill_ctd_table_alr(ctd_table: pd.DataFrame, data_table: pd.DataFrame) -> pd.DataFrame:
    """Fill from an ALR segment table (``TIME``, ``DEPTH_SCI``, optional
    ``SCI_PRESS``/``LAT``/``LON``/``TEMPERATURE``/``CONDUCTIVITY``).  Pressure
    falls back to depth 1:1 when ``SCI_PRESS`` is absent.  Temperature and
    conductivity come from the SBE 52-MP (deg C, mS/cm); oxygen, whose unit
    depends on the sensor set-up, and salinity stay NaN in the UVPapp columns
    (oxygen is still written in the extra ``OXYGEN`` column)."""
    n = len(data_table)
    ctd_table = ctd_table.copy()
    if "qc flag" in ctd_table.columns:
        ctd_table["qc flag"] = np.zeros(n)

    if "TIME" in data_table.columns:
        time_strings = datestr_yyyymmddHHMMSSFFF(epoch_seconds_to_datetime64(data_table["TIME"].to_numpy(dtype=float)))
    else:
        import warnings
        warnings.warn("TIME column not found; leaving time strings empty.")
        time_strings = [""] * n
    if TIME_COLUMN in ctd_table.columns:
        ctd_table[TIME_COLUMN] = time_strings
    elif "time [yyyymmddHHMMSSFFF]" in ctd_table.columns:
        ctd_table["time [yyyymmddHHMMSSFFF]"] = time_strings
    else:
        raise ValueError("CTD table lacks a recognizable time column.")

    if "DEPTH_SCI" in data_table.columns:
        depth_vals = data_table["DEPTH_SCI"].to_numpy(dtype=float)
    else:
        import warnings
        warnings.warn("DEPTH_SCI column not found; setting depth to NaN.")
        depth_vals = np.full(n, np.nan)
    if "depth [m]" in ctd_table.columns:
        ctd_table["depth [m]"] = depth_vals

    if "SCI_PRESS" in data_table.columns:
        press_vals = data_table["SCI_PRESS"].to_numpy(dtype=float)
    else:
        press_vals = depth_vals  # 1 dbar ~ 1 m as requested
    if "pressure [db]" in ctd_table.columns:
        ctd_table["pressure [db]"] = press_vals

    for src, col in (("TEMPERATURE", "temperature [degc]"), ("CONDUCTIVITY", "conductivity [ms cm-1]")):
        if src in data_table.columns and col in ctd_table.columns:
            ctd_table[col] = data_table[src].to_numpy(dtype=float)

    extra_cols = [c for c in data_table.columns if c != "TIME"]
    return pd.concat([ctd_table, data_table[extra_cols].reset_index(drop=True)], axis=1)


def fill_ctd_table_float(ctd_table: pd.DataFrame, data_table: pd.DataFrame) -> pd.DataFrame:
    n = len(data_table)
    ctd_table = ctd_table.copy()
    ctd_table["qc flag"] = np.zeros(n)
    ctd_table[TIME_COLUMN] = datestr_yyyymmddHHMMSSFFF(data_table.iloc[:, 0].to_numpy(dtype="datetime64[ns]"))
    ctd_table["chloro fluo [mg chl m-3]"] = data_table["CHLA_ADJUSTED (mg/m3)"].to_numpy()
    ctd_table["oxygen [umol kg-1]"] = data_table["DOXY_ADJUSTED (µmol/kg)"].to_numpy()
    ctd_table["temperature [degc]"] = data_table["TEMP_ADJUSTED (degC)"].to_numpy()
    ctd_table["pressure [db]"] = data_table["PRES_ADJUSTED (decibar)"].to_numpy()
    ctd_table["practical salinity [psu]"] = data_table["PSAL_ADJUSTED (psu)"].to_numpy()
    return pd.concat([ctd_table, data_table.iloc[:, 1:].reset_index(drop=True)], axis=1)


def fill_ctd_table_seaglider(ctd_table: pd.DataFrame, data_table: pd.DataFrame) -> pd.DataFrame:
    n = len(data_table)
    ctd_table = ctd_table.copy()
    ctd_table["qc flag"] = np.zeros(n)
    ctd_table[TIME_COLUMN] = datestr_yyyymmddHHMMSSFFF(data_table.iloc[:, 0].to_numpy(dtype="datetime64[ns]"))
    ctd_table["conductivity [ms cm-1]"] = data_table["conductivity (S/m)"].to_numpy(dtype=float) * 10
    ctd_table["temperature [degc]"] = data_table["temperature (degrees_Celsius)"].to_numpy()
    ctd_table["pressure [db]"] = data_table["pressure (dbar)"].to_numpy()
    ctd_table["practical salinity [psu]"] = data_table["salinity (1e-3)"].to_numpy(dtype=float) / 1000
    ctd_table["depth [m]"] = data_table["depth (meters)"].to_numpy()
    return pd.concat([ctd_table, data_table.iloc[:, 1:].reset_index(drop=True)], axis=1)


def fill_ctd_table_seaexplorer(ctd_table: pd.DataFrame, data_table: pd.DataFrame) -> pd.DataFrame:
    n = len(data_table)
    ctd_table = ctd_table.copy()
    ctd_table["qc flag"] = np.zeros(n)
    ctd_table[TIME_COLUMN] = datestr_yyyymmddHHMMSSFFF(data_table.iloc[:, 0].to_numpy(dtype="datetime64[ns]"))

    try:
        ctd_table["pressure [db]"] = data_table["LEGATO_PRESSURE"].to_numpy(dtype=float)
    except KeyError:
        ctd_table["pressure [db]"] = data_table["NAV_DEPTH"].to_numpy(dtype=float)

    try:
        ctd_table["chloro fluo [mg chl m-3]"] = data_table["FLBBCD_CHL_SCALED"].to_numpy(dtype=float)
    except KeyError:
        print("No fluorescence data in the glider file")

    try:
        ctd_table["fcdom [ppb qse]"] = data_table["FLBBCD_CDOM_SCALED"].to_numpy(dtype=float)
    except KeyError:
        print("No fcdom data in the glider file")

    try:
        ctd_table["practical salinity [psu]"] = data_table["LEGATO_SALINITY"].to_numpy(dtype=float)
        ctd_table["temperature [degc]"] = data_table["LEGATO_TEMPERATURE"].to_numpy(dtype=float)
        # Computed but not written in the MATLAB original ("correction a faire").
        _oxy = 0.001 * data_table["AROD_FT_DO"].to_numpy(dtype=float) / sw_dens(
            data_table["LEGATO_SALINITY"].to_numpy(dtype=float),
            data_table["LEGATO_TEMPERATURE"].to_numpy(dtype=float),
            data_table["LEGATO_PRESSURE"].to_numpy(dtype=float),
        )
    except KeyError:
        print("No salinity in the glider file")

    try:
        ctd_table["conductivity [ms cm-1]"] = data_table["LEGATO_CONDUCTIVITY"].to_numpy(dtype=float)
    except KeyError:
        print("No conductivity in the glider file")

    # data_table(:,14:21) = []; ctd_table = [ctd_table, data_table(:,3:end-1)]
    data_table = data_table.drop(columns=data_table.columns[13:21])
    return pd.concat([ctd_table, data_table.iloc[:, 2:-1].reset_index(drop=True)], axis=1)


_FILLERS: dict[str, Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]] = {
    "float": fill_ctd_table_float,
    "SeaExplorer": fill_ctd_table_seaexplorer,
    "SeaGlider": fill_ctd_table_seaglider,
    "ALR": fill_ctd_table_alr,
}


# ---------------------------------------------------------------------------
# CreateCTDfile
# ---------------------------------------------------------------------------

def create_ctd_file(project_folder, data_table: pd.DataFrame, filename: str, vector_type: str) -> Path:
    """Write ``<project>/CTDdata/<filename with .ctd>`` from a platform data
    table and return its path.

    MATLAB writes ``.csv`` then renames the last three characters to ``ctd``;
    the file is written directly under its final name here.  ``CTDdata`` is
    created if missing (MATLAB fails silently in that case).
    """
    ctd_table = pd.DataFrame(np.full((len(data_table), len(CTD_COLUMN_NAMES)), np.nan), columns=CTD_COLUMN_NAMES)
    filler = _FILLERS.get(vector_type)
    if filler is not None:
        ctd_table = filler(ctd_table, data_table)

    folder = Path(project_folder) / "CTDdata"
    folder.mkdir(parents=True, exist_ok=True)
    name = str(filename)
    target = folder / (name[:-3] + "ctd")
    write_table_tab(ctd_table, target, encoding="latin-1")
    return target
