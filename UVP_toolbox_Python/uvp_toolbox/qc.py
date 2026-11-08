"""Quality checks of an ALR project, run at the end of the sample creation.

Not a MATLAB translation: these checks were added to the Python port.

    depth_agreement   scale, offset and clock lag between the UVP and ALR depths
    plot_phase_coverage   which ALR phases received UVP images
    plot_sample_sizes     number of images per sample, by phase type

Writes these figures, and a text summary:

    <outdir>/uvp_alr_depth_agreement.png
    <outdir>/uvp_alr_phase_coverage.png
    <outdir>/uvp_alr_sample_sizes.png
    <outdir>/qc_summary.txt
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from .diagnostics import BinnedDepths, bin_depths, _concat_series, _day_floor
from .get_meta_from_vector_meta_file import AlrMatch

__all__ = [
    "DepthAgreement",
    "depth_agreement",
    "plot_depth_agreement",
    "plot_phase_coverage",
    "plot_sample_sizes",
    "run_alr_qc",
]

# Same colours as main/alr_plots.py and the documentation (Table 4.1)
PHASE_COLOURS = {"d": "#2a78d6", "u": "#eb6834", "t": "#1baf7a", "s": "#eda100", "x": "#9a9994"}
PHASE_NAMES = {"d": "Downcast", "u": "Upcast", "t": "Transect", "s": "Surface transect", "x": "Transition"}

# Warning thresholds of the depth agreement
SCALE_TOL_PCT = 0.5     # |scale| above this: the two depths use different units or calibrations
OFFSET_TOL_M = 0.5      # |offset| above this: sensor zero, or distance between the sensors
LAG_TOL_S = 1.0         # |lag| above this: clock offset between the UVP and the ALR
# A depth in metres of fresh water (mH2O) against a pressure in dbar reads this much deeper
MH2O_PER_DBAR_PCT = 1.97


# ---------------------------------------------------------------------------
# 1. Depth agreement: UVP - ALR = scale x depth + offset + lag x vertical velocity
# ---------------------------------------------------------------------------

@dataclass
class DepthAgreement:
    scale_pct: float        # % of depth
    offset_m: float
    lag_s: float            # > 0: the UVP depth is the ALR depth lag_s seconds later
    rms_m: float            # what the model leaves unexplained
    n_bins: int
    days: list[dict] = field(default_factory=list)   # the same, per day

    def warnings(self) -> list[str]:
        out = []
        if abs(self.scale_pct) > SCALE_TOL_PCT:
            hint = (f" (a mH2O vs dbar unit difference gives +{MH2O_PER_DBAR_PCT} %)"
                    if abs(self.scale_pct - MH2O_PER_DBAR_PCT) < 0.3 else "")
            out.append(f"UVP depth differs from ALR depth by {self.scale_pct:+.2f} % of depth{hint}")
        if abs(self.offset_m) > OFFSET_TOL_M:
            out.append(f"constant depth offset of {self.offset_m:+.2f} m between UVP and ALR")
        if abs(self.lag_s) > LAG_TOL_S:
            out.append(f"clock lag of {self.lag_s:+.1f} s between UVP and ALR")
        return out


def _fit(z, d, w):
    """Least squares of d = a z + b + lag w; returns (a, b, lag, rms) or None."""
    ok = np.isfinite(z) & np.isfinite(d) & np.isfinite(w)
    if ok.sum() < 10 or np.ptp(z[ok]) < 10:
        return None
    X = np.c_[z[ok], np.ones(ok.sum()), w[ok]]
    coef, *_ = np.linalg.lstsq(X, d[ok], rcond=None)
    rms = float(np.sqrt(np.mean((d[ok] - X @ coef) ** 2)))
    return float(coef[0]), float(coef[1]), float(coef[2]), rms


def depth_agreement(binned: BinnedDepths, bin_seconds: float = 10.0) -> DepthAgreement | None:
    """Fit UVP - ALR = scale x depth + offset + lag x w on the 10 s median bins.

    w is the vertical velocity of the ALR depth: a clock lag shifts the two
    depth records in time, which shows as a difference proportional to w,
    opposite on downcasts and upcasts. The lag is resolved well below the
    10 s bin size because it is fitted, not searched bin by bin.
    """
    if binned.empty:
        return None
    z, d = binned.alr_med, binned.uvp_med - binned.alr_med
    w = np.gradient(binned.alr_med, bin_seconds)
    whole = _fit(z, d, w)
    if whole is None:
        return None
    days = []
    day = _day_floor(binned.centers)
    for ds in np.unique(day[~np.isnat(day)]):
        m = day == ds
        res = _fit(z[m], d[m], w[m])
        if res is not None:
            days.append({"day": ds, "scale_pct": 100 * res[0], "offset_m": res[1], "lag_s": res[2],
                         "n_bins": int((np.isfinite(z[m]) & np.isfinite(d[m])).sum())})
    n = int((np.isfinite(z) & np.isfinite(d) & np.isfinite(w)).sum())
    return DepthAgreement(100 * whole[0], whole[1], whole[2], whole[3], n, days)


def plot_depth_agreement(out: Path, binned: BinnedDepths, agr: DepthAgreement) -> Path:
    z, d = binned.alr_med, binned.uvp_med - binned.alr_med
    ok = np.isfinite(z) & np.isfinite(d)
    fig = plt.figure(figsize=(13, 7), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[3, 2])
    ax = fig.add_subplot(gs[:, 0])
    ax.plot(z[ok], d[ok], ".", ms=1.5, color="0.6", rasterized=True)
    zz = np.array([0, np.nanmax(z[ok])])
    ax.plot(zz, agr.scale_pct / 100 * zz + agr.offset_m, color=PHASE_COLOURS["d"], lw=1.5,
            label=f"fit: {agr.scale_pct:+.2f} % × depth {agr.offset_m:+.2f} m")
    ax.plot(zz, MH2O_PER_DBAR_PCT / 100 * zz, color="0.3", ls="--", lw=1,
            label=f"mH2O vs dbar alone: +{MH2O_PER_DBAR_PCT} %")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.grid(True, color="0.9")
    ax.set_xlabel("ALR depth (m)")
    ax.set_ylabel("UVP − ALR depth (m), 10 s medians")
    ax.legend(loc="upper left", frameon=False)
    warn = agr.warnings()
    if warn:
        ax.text(0.02, 0.84, "\n".join("WARNING: " + w.replace(" (", "\n    (") for w in warn), transform=ax.transAxes,
                ha="left", va="top", fontsize=9, color="#b3261e",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#b3261e", lw=0.8), wrap=True)
    ax.set_title(f"Depth agreement: scale {agr.scale_pct:+.2f} %, offset {agr.offset_m:+.2f} m, "
                 f"clock lag {agr.lag_s:+.1f} s ({agr.n_bins:,} bins)", loc="left")

    days = agr.days
    t = np.array([np.datetime64(r["day"], "D") for r in days])
    for k, (key, label, tol) in enumerate((("scale_pct", "Scale (%)", SCALE_TOL_PCT),
                                           ("offset_m", "Offset (m)", OFFSET_TOL_M),
                                           ("lag_s", "Clock lag (s)", LAG_TOL_S))):
        a = fig.add_subplot(gs[k, 1])
        v = np.array([r[key] for r in days])
        a.axhspan(-tol, tol, color="0.92", zorder=0)
        a.axhline(0, color="0.4", lw=0.8)
        a.plot(t, v, "o", ms=4, color=PHASE_COLOURS["d"])
        a.set_ylabel(label)
        a.grid(True, color="0.9")
        locator = mdates.AutoDateLocator(maxticks=6)
        a.xaxis.set_major_locator(locator)
        a.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        if k == 0:
            a.set_title(f"Per day, {len(days)} days (grey band: within tolerance)", loc="left")
    fig.suptitle("QC – UVP vs ALR depth – " + (f"{len(warn)} warning{'s' if len(warn) > 1 else ''}" if warn else "OK"),
                 x=0.01, ha="left", fontweight="bold", color="#b3261e" if warn else "black")
    fig.canvas.manager.set_window_title("QC – UVP vs ALR depth")
    path = out / "uvp_alr_depth_agreement.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 2. Phase coverage: which ALR phases received UVP images
# ---------------------------------------------------------------------------

def plot_phase_coverage(out: Path, phases: Sequence[dict]) -> tuple[Path, str]:
    """One row per phase type, one bar per ALR phase: coloured when it has UVP
    images, grey with a red outline when it has none."""
    rows = [p for p in phases if p["start"] is not None]
    n_all, n_miss = len(phases), sum(1 for p in phases if p["n_images"] == 0)
    by_type = Counter(p["letter"] for p in phases if p["n_images"] == 0)
    fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
    letters = list(PHASE_COLOURS)
    for i, k in enumerate(letters):
        mine = [p for p in rows if p["letter"] == k]
        x0 = mdates.date2num(np.array([p["start"] for p in mine], dtype="datetime64[ns]"))
        x1 = mdates.date2num(np.array([p["end"] for p in mine], dtype="datetime64[ns]"))
        has = np.array([p["n_images"] > 0 for p in mine], dtype=bool)
        width = np.maximum(x1 - x0, 1 / 1440)  # at least one minute, so short phases stay visible
        if has.any():
            ax.broken_barh(list(zip(x0[has], width[has])), (i - 0.4, 0.8), facecolors=PHASE_COLOURS[k])
        if (~has).any():
            ax.broken_barh(list(zip(x0[~has], width[~has])), (i - 0.4, 0.8),
                           facecolors="0.85", edgecolors="#b3261e", linewidth=0.6)
    ax.set_yticks(range(len(letters)), [f"{k} – {PHASE_NAMES[k]}" for k in letters])
    ax.invert_yaxis()
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.grid(True, axis="x", color="0.9")
    detail = ", ".join(f"{by_type[k]} {k}" for k in letters if by_type[k])
    summary = f"{n_all - n_miss} of {n_all} ALR phases have UVP images; {n_miss} have none" + (f" ({detail})" if detail else "")
    ax.set_title(f"QC – phase coverage: {summary}\nColoured: phase with UVP images; grey with red outline: no UVP image",
                 loc="left")
    fig.canvas.manager.set_window_title("QC – phase coverage")
    path = out / "uvp_alr_phase_coverage.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, summary


# ---------------------------------------------------------------------------
# 3. Sample sizes: number of images per sample, by phase type
# ---------------------------------------------------------------------------

def plot_sample_sizes(out: Path, samples: Sequence[tuple[str, str, int]], min_images: int = 20) -> tuple[Path, str]:
    """``samples`` = (sample name, phase letter, number of images)."""
    n = np.array([s[2] for s in samples])
    small = [s for s in samples if s[2] < min_images]
    by_type = Counter(s[1] for s in small)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    bins = np.logspace(0, np.log10(max(n.max(), 10)) + 0.1, 40)
    for k, colour in PHASE_COLOURS.items():
        v = [s[2] for s in samples if s[1] == k]
        if v:
            ax.hist(v, bins=bins, histtype="step", lw=1.6, color=colour, label=f"{k} – {PHASE_NAMES[k]} ({len(v)})")
    ax.axvline(min_images, color="#b3261e", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Images per sample")
    ax.set_ylabel("Samples")
    ax.grid(True, color="0.9")
    ax.legend(frameon=False)
    detail = ", ".join(f"{by_type[k]} {k}" for k in PHASE_COLOURS if by_type[k])
    summary = f"{len(small)} of {len(samples)} samples have fewer than {min_images} images" + (f" ({detail})" if detail else "")
    ax.set_title(f"QC – sample sizes: {summary} (red line)", loc="left")
    fig.canvas.manager.set_window_title("QC – sample sizes")
    path = out / "uvp_alr_sample_sizes.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, summary


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_alr_qc(outdir, uvp_time_series: Sequence, uvp_depth_series: Sequence, match: AlrMatch,
               samples: Sequence[tuple[str, str, int]], min_images: int = 20, verbose: bool = True) -> dict:
    """Run the three checks, write their figures and ``qc_summary.txt``, print
    the summary, and return it with the figure paths."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    lines, written = [], {}

    t_uvp, z_uvp = _concat_series(uvp_time_series, uvp_depth_series)
    binned = bin_depths(t_uvp, z_uvp, match.alr_plot_time, match.alr_plot_depth)
    agr = depth_agreement(binned)
    if agr is None:
        lines.append("Depth agreement: not enough joint UVP/ALR data with a depth range")
    else:
        written["depth_agreement"] = plot_depth_agreement(out, binned, agr)
        lines.append(f"Depth agreement: UVP - ALR = {agr.scale_pct:+.2f} % x depth {agr.offset_m:+.2f} m, "
                     f"clock lag {agr.lag_s:+.1f} s (rms {agr.rms_m:.2f} m, {agr.n_bins} bins)")
        lines += [f"  WARNING: {w}" for w in agr.warnings()]

    if match.phases:
        written["phase_coverage"], summary = plot_phase_coverage(out, match.phases)
        lines.append(f"Phase coverage: {summary}")
    if samples:
        written["sample_sizes"], summary = plot_sample_sizes(out, samples, min_images)
        lines.append(f"Sample sizes: {summary}")
        smallest = sorted(samples, key=lambda s: s[2])[:5]
        lines.append("  smallest: " + ", ".join(f"{s[0]} ({s[2]})" for s in smallest))

    (out / "qc_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if verbose:
        print("\n--- Quality checks (also in qc_summary.txt) ---")
        print("\n".join(lines))
    return {"lines": lines, "figures": written, "depth_agreement": agr}
