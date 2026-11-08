"""Readers for the vehicle ("vector") metadata files.

    ReadMetaALR              -> read_meta_alr
    ReadMetaSeaglider        -> read_meta_seaglider
    ReadMetaFloat            -> read_meta_float
    ReadDataSeaexplorer      -> read_data_seaexplorer
    ConvertLatLonSeaexplorer -> convert_latlon_seaexplorer

Every reader returns ``(VectorMeta, data_table)``.  The MATLAB ``meta`` is an
``N x 4`` numeric array ``[datenum, depth, lat, lon]``; here it is a small
dataclass whose ``time`` is ``datetime64[ns]``.  The MATLAB ``table`` becomes a
``pandas.DataFrame`` with the same column names, except that time columns hold
``datetime64`` values instead of datenums.

Original MATLAB script by Camille Catalano, LOV, 2020/11
"""

from __future__ import annotations

import gzip
import io
import warnings
from dataclasses import dataclass
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

from .matlab_compat import epoch_seconds_to_datetime64

__all__ = [
    "VectorMeta",
    "ncread",
    "read_meta_alr",
    "read_meta_seaglider",
    "read_meta_float",
    "read_data_seaexplorer",
    "convert_latlon_seaexplorer",
]


@dataclass
class VectorMeta:
    """The MATLAB ``meta`` array: columns 1, 2, 3, 4 = time, depth, lat, lon."""
    time: np.ndarray   # datetime64[ns]
    depth: np.ndarray  # float, NaN where the platform has none
    lat: np.ndarray
    lon: np.ndarray

    def __len__(self) -> int:
        return int(self.time.size)


def ncread(ds: netCDF4.Dataset, name: str) -> np.ndarray:
    """MATLAB ``ncread``: scale/offset applied, fill values -> NaN, and
    dimensions in MATLAB (column-major) order."""
    var = ds.variables[name]
    a = var[:]
    if np.ma.isMaskedArray(a):
        a = a.astype(float).filled(np.nan) if a.dtype.kind in "iufb" else a.filled()
    a = np.asarray(a)
    if a.dtype.kind in "iub":
        a = a.astype(float)
    if a.ndim > 1:
        a = a.T
    return a


# ---------------------------------------------------------------------------
# ReadMetaALR
# ---------------------------------------------------------------------------

def read_meta_alr(nc_path) -> tuple[VectorMeta, pd.DataFrame]:
    """Read an ALR segment netCDF (as written by ``main/alr_data.py``).

    ``data_table`` columns: ``TIME`` (s since 1970-01-01 UTC), ``DEPTH_SCI``,
    and ``SCI_PRESS`` / ``LAT`` / ``LON`` / ``TEMPERATURE`` / ``CONDUCTIVITY`` /
    ``OXYGEN`` / ``PITCH_ENG`` / ``ROLL_ENG`` when present.
    """
    with netCDF4.Dataset(nc_path, "r") as ds:
        time_s = ncread(ds, "TIME").ravel()
        depth_sci = ncread(ds, "DEPTH_SCI").ravel()
        optional: dict[str, np.ndarray] = {}
        for name in ("SCI_PRESS", "LAT", "LON", "TEMPERATURE", "CONDUCTIVITY", "OXYGEN", "PITCH_ENG", "ROLL_ENG"):
            if name in ds.variables:
                optional[name] = ncread(ds, name).ravel()
    cols: dict[str, np.ndarray] = {"TIME": time_s, "DEPTH_SCI": depth_sci}
    cols.update(optional)
    data_table = pd.DataFrame(cols)
    n = time_s.size
    meta = VectorMeta(
        time=epoch_seconds_to_datetime64(time_s),
        depth=np.full(n, np.nan),
        lat=optional.get("LAT", np.full(n, np.nan)),
        lon=optional.get("LON", np.full(n, np.nan)),
    )
    return meta, data_table


# ---------------------------------------------------------------------------
# ReadMetaSeaglider
# ---------------------------------------------------------------------------

def read_meta_seaglider(nc_path) -> tuple[VectorMeta, pd.DataFrame]:
    with netCDF4.Dataset(nc_path, "r") as ds:
        time = epoch_seconds_to_datetime64(ncread(ds, "time").ravel())
        depth = ncread(ds, "depth").ravel()
        try:
            latitude = ncread(ds, "latitude").ravel()
            longitude = ncread(ds, "longitude").ravel()
        except KeyError:
            warnings.warn(f"No lat-lon in file {nc_path}")
            latitude = np.zeros(time.size)
            longitude = np.zeros(time.size)
        data_table = pd.DataFrame({
            "time (datenum)": time,
            "pressure (dbar)": ncread(ds, "pressure").ravel(),
            "depth (meters)": depth,
            "temperature (degrees_Celsius)": ncread(ds, "temperature").ravel(),
            "conductivity (S/m)": ncread(ds, "conductivity").ravel(),
            "salinity (1e-3)": ncread(ds, "salinity").ravel(),
            "sound_velocity (m/s)": ncread(ds, "sound_velocity").ravel(),
            "latitude (degrees_north)": latitude,
            "longitude (degrees_east)": longitude,
        })
    return VectorMeta(time=time, depth=depth, lat=latitude, lon=longitude), data_table


# ---------------------------------------------------------------------------
# ReadMetaFloat
# ---------------------------------------------------------------------------

_ARGO_EPOCH_NS = np.datetime64("1950-01-01T00:00:00", "ns")


def _single_profile(a: np.ndarray, name: str) -> np.ndarray:
    """Argo profile variables are ``N_LEVELS x N_PROF`` in MATLAB order.  The
    MATLAB code only works for a single profile per file (its ``depth*0 + juld``
    broadcast fails otherwise), so that is what is supported here."""
    a = np.asarray(a)
    if a.ndim == 2:
        if a.shape[1] != 1:
            raise NotImplementedError(
                f"{name}: {a.shape[1]} profiles in one file; the toolbox expects synthetic files with N_PROF == 1")
        a = a[:, 0]
    return a.ravel()


def _optional_argo(ds: netCDF4.Dataset, name: str, like: np.ndarray) -> np.ndarray:
    try:
        return _single_profile(ncread(ds, name), name)
    except KeyError:
        return np.full(like.shape, np.nan)


def read_meta_float(nc_path) -> tuple[VectorMeta, pd.DataFrame]:
    with netCDF4.Dataset(nc_path, "r") as ds:
        depth = _single_profile(ncread(ds, "PRES_ADJUSTED"), "PRES_ADJUSTED")
        if np.all(np.isnan(depth)):  # prod(isnan(depth))
            depth = _single_profile(ncread(ds, "PRES"), "PRES")
        juld = np.full(depth.shape, float(np.ravel(ncread(ds, "JULD"))[0]))
        time = _ARGO_EPOCH_NS + np.round(juld * 86400.0 * 1e9).astype(np.int64).astype("timedelta64[ns]")
        latitude = np.full(depth.shape, float(np.ravel(ncread(ds, "LATITUDE"))[0]))
        longitude = np.full(depth.shape, float(np.ravel(ncread(ds, "LONGITUDE"))[0]))
        temp = _single_profile(ncread(ds, "TEMP_ADJUSTED"), "TEMP_ADJUSTED")
        cycle_nb = np.full(depth.shape, float(np.ravel(ncread(ds, "CYCLE_NUMBER"))[0]))
        bbp700 = _optional_argo(ds, "BBP700_ADJUSTED", depth)
        chla = _optional_argo(ds, "CHLA_ADJUSTED", depth)
        doxy = _optional_argo(ds, "DOXY_ADJUSTED", depth)
        psal = _optional_argo(ds, "PSAL_ADJUSTED", depth)

    data_table = pd.DataFrame({
        "time (datenum)": time,
        "JULD (days since 1950-01-01 00:00:00 UTC)": juld,
        "PRES_ADJUSTED (decibar)": depth,
        "LATITUDE (degree north)": latitude,
        "LONGITUDE (degree east)": longitude,
        "BBP700_ADJUSTED (m-1)": bbp700,
        "CHLA_ADJUSTED (mg/m3)": chla,
        "DOXY_ADJUSTED (µmol/kg)": doxy,
        "PSAL_ADJUSTED (psu)": psal,
        "TEMP_ADJUSTED (degC)": temp,
        "CYCLE_NUMBER": cycle_nb,
    })
    return VectorMeta(time=time, depth=depth, lat=latitude, lon=longitude), data_table


# ---------------------------------------------------------------------------
# ReadDataSeaexplorer / ConvertLatLonSeaexplorer
# ---------------------------------------------------------------------------

def read_data_seaexplorer(gz_path) -> tuple[VectorMeta, pd.DataFrame]:
    """Read a gzipped SeaExplorer ``*.pld1.raw.N.gz`` payload log.

    The archive is decompressed in memory (MATLAB gunzips to disk and deletes
    the copy afterwards).
    """
    with gzip.open(gz_path, "rt", encoding="latin-1") as f:
        text = f.read()
    first_line = text.split("\n", 1)[0]
    if ";" in first_line:
        df = pd.read_csv(io.StringIO(text), sep=";")
    else:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    df["PLD_REALTIMECLOCK"] = pd.to_datetime(df["PLD_REALTIMECLOCK"], format="%d/%m/%Y %H:%M:%S.%f", errors="coerce")
    for col in ("NAV_DEPTH", "NAV_LATITUDE", "NAV_LONGITUDE"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    time = df["PLD_REALTIMECLOCK"].to_numpy(dtype="datetime64[ns]")
    meta = VectorMeta(
        time=time,
        depth=df["NAV_DEPTH"].to_numpy(dtype=float),
        lat=df["NAV_LATITUDE"].to_numpy(dtype=float),
        lon=df["NAV_LONGITUDE"].to_numpy(dtype=float),
    )
    return meta, df


def convert_latlon_seaexplorer(latlon):
    """``DDMM.MMM`` -> decimal degrees (``fix(x/100) + rem(x,100)/60``)."""
    x = np.asarray(latlon, dtype=float)
    return np.trunc(x / 100.0) + np.fmod(x, 100.0) / 60.0
