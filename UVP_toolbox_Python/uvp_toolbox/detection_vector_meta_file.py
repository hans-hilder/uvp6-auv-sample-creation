"""Locate the vehicle metadata folder inside a UVP project (``meta`` for ALR, ``doc`` otherwise).

Translation of ``UVP6_samples_creation/DetectionVectorMetaFile.m``.

Original MATLAB script by Camille Catalano, LOV, 2020/11
"""

from __future__ import annotations

import gzip
import re
import shutil
import warnings
from pathlib import Path

import netCDF4

__all__ = ["detection_vector_meta_file"]


def detection_vector_meta_file(project_folder, vector_type: str) -> tuple[Path, str]:
    """Return ``(meta_data_folder, vector_sn)`` for the project.

    ``vector_sn`` is the platform label written to the ``ship`` column of the
    sample file (``Seaeplorer_###``, ``SeaGlider_###``, ``WMO#######`` or
    ``ALR###``).
    """
    doc = Path(project_folder) / "doc"

    if vector_type == "SeaExplorer":
        hits = sorted(doc.glob("SEA*"))
        if not hits:
            raise FileNotFoundError("ERROR : No metadata folder found in \\doc")
        vector_sn = "Seaeplorer_" + hits[0].name[3:6]
    elif vector_type == "SeaGlider":
        hits = sorted(doc.glob("SG*"))
        if not hits:
            raise FileNotFoundError("ERROR : No metadata folder found in \\doc")
        vector_sn = "SeaGlider_" + hits[0].name[2:5]
    elif vector_type == "float":
        hits = sorted(doc.glob("float*"))
        if not hits:
            raise FileNotFoundError("ERROR : No metadata folder found in \\doc")
        vector_sn = "WMO" + hits[0].name[-7:]
    elif vector_type == "ALR":
        # meta/ since #11; doc/ is still read for projects processed before
        hits = sorted((Path(project_folder) / "meta").glob("alr_meta_sn*")) or sorted(doc.glob("alr_meta_sn*"))
        if not hits:
            raise FileNotFoundError('ERROR : No ALR metadata folder found in \\meta (expected "alr_meta_sn*")')
        m = re.search(r"sn(\d+)", hits[0].name)
        if m:
            vector_sn = "ALR%03d" % int(m.group(1))
        else:
            nc_list = sorted(hits[0].glob("*.nc"))
            if not nc_list:
                raise FileNotFoundError("ERROR: ALR meta folder contains no .nc files")
            try:
                with netCDF4.Dataset(nc_list[0], "r") as ds:
                    vector_sn = re.sub(r"[^\w]", "", str(ds.getncattr("alr_serial")))
            except Exception:
                warnings.warn('Could not read alr_serial; using "ALR".')
                vector_sn = "ALR"
    else:
        raise ValueError(f"Unknown vector type {vector_type!r}")

    first = hits[0]
    meta_data_folder = first
    if not first.is_dir():
        # MATLAB: gunzip(meta_data_folder, folder); meta_data_folder = name(1:end-4).
        # Ported literally, including the four-character strip.
        target = first.parent / (first.name[:-3] if first.name.endswith(".gz") else first.name + ".out")
        with gzip.open(first, "rb") as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
        meta_data_folder = first.parent / first.name[:-4]
        if not meta_data_folder.exists():
            warnings.warn(f"Decompressed {first.name} but {meta_data_folder} does not exist (literal port of the MATLAB name handling)")

    print(f"Vector meta data folder : {first.name}")
    return meta_data_folder, vector_sn
