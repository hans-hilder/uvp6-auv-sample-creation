"""Quality checks of an ALR project, run at the end of the sample creation.

Not a MATLAB translation: these checks were added to the Python port.

    depth_agreement     scale, offset and clock lag between the UVP and ALR depths
    coverage_gaps       when and where the ALR ran without UVP images
    plot_coverage_map   the ALR track with those gaps, and the phases on a time line
    plot_sample_sizes   number of images per sample, by phase type

Writes these figures, a table of the gaps, and a text summary:

    <outdir>/uvp_alr_depth_agreement.png
    <outdir>/uvp_alr_coverage_map.png
    <outdir>/uvp_alr_sample_sizes.png
    <outdir>/qc_coverage.csv
    <outdir>/qc_summary.txt
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from .diagnostics import BinnedDepths, bin_depths, _concat_series, _day_floor
from .get_meta_from_vector_meta_file import AlrMatch
from .matlab_compat import finite_times

__all__ = [
    "DepthAgreement",
    "depth_agreement",
    "plot_depth_agreement",
    "CoverageGap",
    "coverage_gaps",
    "coverage_table",
    "plot_coverage_map",
    "plot_sample_sizes",
    "run_alr_qc",
]

# Same colours as main/alr_plots.py and the documentation (Table 4.1)
PHASE_COLOURS = {"d": "#2a78d6", "u": "#eb6834", "t": "#1baf7a", "s": "#eda100", "x": "#5f5e5a"}
PHASE_NAMES = {"d": "Downcast", "u": "Upcast", "t": "Transect", "s": "Surface transect", "x": "Transition"}

# Warning thresholds of the depth agreement
SCALE_TOL_PCT = 0.5     # |scale| above this: the two depths use different units or calibrations
OFFSET_TOL_M = 0.5      # |offset| above this: sensor zero, or distance between the sensors
LAG_TOL_S = 1.0         # |lag| above this: clock offset between the UVP and the ALR
# A depth in metres of fresh water (mH2O) against a pressure in dbar reads this much deeper
MH2O_PER_DBAR_PCT = 1.97
# Consecutive phases without UVP image are listed as a gap from this duration on
MIN_GAP_HOURS = 1.0
C_GAP = "#b3261e"


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
# 2. Coverage: when and where the ALR ran without UVP images
# ---------------------------------------------------------------------------

def _km(lat, lon) -> float:
    """Length (km) of a lat/lon track, great circles between points."""
    lat, lon = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    if lat.size < 2:
        return 0.0
    a = (np.sin(np.diff(lat) / 2) ** 2
         + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(np.diff(lon) / 2) ** 2)
    return float(6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))).sum())


def _track(phases: Sequence[dict]):
    """Time, lat, lon of the phases' GPS tracks, in time order."""
    parts = [p for p in phases if p.get("track_time") is not None and len(p["track_time"])]
    if not parts:
        return np.array([], "datetime64[ns]"), np.array([]), np.array([])
    t = np.concatenate([p["track_time"] for p in parts])
    order = np.argsort(t, kind="stable")
    return (t[order], np.concatenate([p["track_lat"] for p in parts])[order],
            np.concatenate([p["track_lon"] for p in parts])[order])


@dataclass
class CoverageGap:
    """A run of consecutive ALR phases without UVP image, from the last UVP
    image before it to the first one after it."""
    start: np.datetime64
    end: np.datetime64
    phases: list[dict]
    lat: np.ndarray         # ALR track from start to end, both ends interpolated
    lon: np.ndarray

    @property
    def hours(self) -> float:
        return float((self.end - self.start) / np.timedelta64(1, "s")) / 3600

    def row(self, k: int) -> dict:
        """One line of qc_coverage.csv."""
        pos = lambda v, i: f"{v[i]:.4f}" if v.size else ""
        n = Counter(p["letter"] for p in self.phases)
        return {"gap": f"G{k}", "start": str(self.start)[:19], "end": str(self.end)[:19],
                "duration_h": f"{self.hours:.1f}", "start_lat": pos(self.lat, 0), "start_lon": pos(self.lon, 0),
                "end_lat": pos(self.lat, -1), "end_lon": pos(self.lon, -1),
                "distance_km": f"{_km(self.lat, self.lon):.1f}",
                "phases": len(self.phases), **{c: n[c] for c in PHASE_COLOURS}}


def coverage_gaps(phases: Sequence[dict], uvp_times=None, min_hours: float = MIN_GAP_HOURS
                  ) -> tuple[list[CoverageGap], list[CoverageGap]]:
    """Group the time-ordered phases without UVP image into runs; return the
    runs of at least ``min_hours`` (the gaps) and the shorter ones apart: a
    short transition without image is not a loss of data.

    With ``uvp_times`` (every UVP image time), a run is widened to the last
    image before it and the first image after it, as the images can stop or
    start again inside a phase that has some.
    """
    rows = sorted((p for p in phases if p["start"] is not None), key=lambda p: p["start"])
    runs, cur = [], []
    for p in rows:
        if p["n_images"] == 0:
            cur.append(p)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)

    uvp = np.sort(finite_times(np.asarray(uvp_times, dtype="datetime64[ns]").ravel())) if uvp_times is not None else None
    t_all, lat_all, lon_all = _track(rows)
    sec = lambda t: (np.asarray(t, dtype="datetime64[ns]") - np.datetime64(0, "ns")) / np.timedelta64(1, "s")
    gaps = []
    for r in runs:
        start, end = r[0]["start"], r[-1]["end"]
        if uvp is not None and uvp.size:
            i = np.searchsorted(uvp, start)
            if i > 0:
                start = uvp[i - 1]
            j = np.searchsorted(uvp, end, side="right")
            if j < uvp.size:
                end = uvp[j]
        inside = (t_all > start) & (t_all < end)
        if t_all.size:
            ends = sec(np.array([start, end], dtype="datetime64[ns]"))
            lat = np.r_[np.interp(ends[0], sec(t_all), lat_all), lat_all[inside], np.interp(ends[1], sec(t_all), lat_all)]
            lon = np.r_[np.interp(ends[0], sec(t_all), lon_all), lon_all[inside], np.interp(ends[1], sec(t_all), lon_all)]
        else:
            lat = lon = np.array([])
        gaps.append(CoverageGap(start, end, r, lat, lon))
    return [g for g in gaps if g.hours >= min_hours], [g for g in gaps if g.hours < min_hours]


def coverage_table(phases: Sequence[dict], gaps: Sequence[CoverageGap], short: Sequence[CoverageGap],
                   min_hours: float = MIN_GAP_HOURS) -> list[str]:
    """The deployment, the phases by type and the gaps, as lines of qc_summary.txt."""
    rows = [p for p in phases if p["start"] is not None]
    t0, t1 = min(p["start"] for p in rows), max(p["end"] for p in rows)
    days = float((t1 - t0) / np.timedelta64(1, "s")) / 86400
    _, lat, lon = _track(rows)
    lost = sum(g.hours for g in gaps)
    out = [f"Deployment: {str(t0)[:16]} to {str(t1)[:16]} ({days:.1f} days, {_km(lat, lon):,.0f} km of track)",
           f"  {'phase':<22}{'phases':>8}{'with images':>13}{'images':>11}"]
    for k in PHASE_COLOURS:
        mine = [p for p in rows if p["letter"] == k]
        out.append(f"  {k + ' - ' + PHASE_NAMES[k]:<22}{len(mine):>8}{sum(p['n_images'] > 0 for p in mine):>13}"
                   f"{sum(p['n_images'] for p in mine):>11,}")
    out.append(f"  {'all':<22}{len(rows):>8}{sum(p['n_images'] > 0 for p in rows):>13}"
               f"{sum(p['n_images'] for p in rows):>11,}")
    out.append(f"Coverage gaps (>= {min_hours:g} h without UVP image): {len(gaps)}, {lost:.1f} h in all "
               f"({100 * lost / 24 / days:.0f} % of the deployment)" + (", listed in qc_coverage.csv" if gaps else ""))
    for k, g in enumerate(gaps, start=1):
        r = g.row(k)
        out.append(f"  {r['gap']}  {r['start'][:16]} to {r['end'][:16]}  {r['duration_h']} h  "
                   f"from {r['start_lat']}, {r['start_lon']} to {r['end_lat']}, {r['end_lon']}  "
                   f"{r['distance_km']} km  {r['phases']} phases")
    if short:
        n = Counter(p["letter"] for g in short for p in g.phases)
        out.append(f"  and {sum(len(g.phases) for g in short)} phase(s) without image in shorter runs, not gaps ("
                   + ", ".join(f"{n[k]} {k}" for k in PHASE_COLOURS if n[k]) + ")")
    return out


def write_coverage_csv(out: Path, gaps: Sequence[CoverageGap]) -> Path:
    """qc_coverage.csv: one row per gap (header only when there is none)."""
    path = out / "qc_coverage.csv"
    fields = ["gap", "start", "end", "duration_h", "start_lat", "start_lon", "end_lat", "end_lon",
              "distance_km", "phases", *PHASE_COLOURS]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(g.row(k) for k, g in enumerate(gaps, start=1))
    return path


def plot_coverage_map(out: Path, phases: Sequence[dict], gaps: Sequence[CoverageGap],
                      min_hours: float = MIN_GAP_HOURS) -> tuple[Path, str]:
    """The ALR track seen from above, coloured by phase type where the UVP has
    images and red where it has none; below, every phase on a time line."""
    rows = [p for p in phases if p["start"] is not None]
    n_miss = sum(p["n_images"] == 0 for p in rows)
    lost = sum(g.hours for g in gaps)
    summary = (f"{len(rows) - n_miss} of {len(rows)} ALR phases have UVP images; "
               f"{len(gaps)} gap{'s' if len(gaps) != 1 else ''} of >= {min_hours:g} h without image ({lost:.1f} h)")

    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])
    ax = fig.add_subplot(gs[0])
    for p in rows:
        if p.get("track_lat") is None or not len(p["track_lat"]):
            continue
        if p["n_images"] > 0:
            ax.plot(p["track_lon"], p["track_lat"], "-", lw=1, color=PHASE_COLOURS[p["letter"]], zorder=2)
        else:
            ax.plot(p["track_lon"], p["track_lat"], "-", lw=2.5, color=C_GAP, zorder=3)
    t, lat, lon = _track(rows)
    if lat.size:
        ax.plot(lon[0], lat[0], "^", ms=10, color="black", zorder=5)
        ax.annotate(f"start {str(t[0])[:10]}", (lon[0], lat[0]), xytext=(8, -12), textcoords="offset points", fontsize=9)
        ax.plot(lon[-1], lat[-1], "s", ms=9, color="black", zorder=5)
        ax.annotate(f"end {str(t[-1])[:10]}", (lon[-1], lat[-1]), xytext=(8, 4), textcoords="offset points", fontsize=9)
        ax.set_aspect(1 / np.cos(np.radians(np.nanmean(lat))))  # metres east = metres north
    when = lambda t: np.datetime64(t, "s").item().strftime("%d %b %H:%M")
    for k, g in enumerate(gaps, start=1):
        if g.lat.size:
            ax.plot(g.lon, g.lat, "-", lw=2.5, color=C_GAP, zorder=3)
            # open circle where the UVP images stop, filled where they start again
            for i, mfc, label, dy in ((0, "white", f"G{k} from {when(g.start)}", 8), (-1, C_GAP, f"G{k} to {when(g.end)}", -16)):
                ax.plot(g.lon[i], g.lat[i], "o", ms=7, mfc=mfc, mec=C_GAP, mew=1.5, zorder=6)
                ax.annotate(label, (g.lon[i], g.lat[i]), xytext=(-10, dy), textcoords="offset points", ha="right",
                            fontsize=9, color=C_GAP, fontweight="bold", zorder=7,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    handles = [plt.Line2D([], [], color=PHASE_COLOURS[k], lw=2, label=f"{k} – {PHASE_NAMES[k]}") for k in PHASE_COLOURS]
    if n_miss:
        handles.append(plt.Line2D([], [], color=C_GAP, lw=2.5, label="no UVP image"))
    ax.legend(handles=handles, loc="best", fontsize=9)
    ax.grid(True, color="0.9")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    fig.suptitle(f"QC – UVP coverage: {summary}", x=0.01, ha="left", fontweight="bold",
                 color=C_GAP if gaps else "black")

    tl = fig.add_subplot(gs[1])
    x0 = mdates.date2num(np.array([p["start"] for p in rows], dtype="datetime64[ns]"))
    x1 = mdates.date2num(np.array([p["end"] for p in rows], dtype="datetime64[ns]"))
    width = np.maximum(x1 - x0, 1 / 1440)  # at least one minute, so short phases stay visible
    colours = [PHASE_COLOURS[p["letter"]] if p["n_images"] > 0 else C_GAP for p in rows]
    tl.broken_barh(list(zip(x0, width)), (0, 1), facecolors=colours)
    for k, g in enumerate(gaps, start=1):
        # the gap itself, from the last UVP image to the next one, above the phases
        g0, g1 = mdates.date2num(np.array([g.start, g.end], dtype="datetime64[ns]"))
        tl.broken_barh([(g0, g1 - g0)], (1.08, 0.12), facecolors=C_GAP)
        tl.annotate(f"G{k}", ((g0 + g1) / 2, 1.2), xytext=(0, 2), textcoords="offset points", ha="center",
                    va="bottom", fontsize=9, color=C_GAP, fontweight="bold")
    tl.set_ylim(0, 1.6)
    tl.set_yticks([])
    locator = mdates.AutoDateLocator()
    tl.xaxis.set_major_locator(locator)
    tl.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    tl.grid(True, axis="x", color="0.9")
    tl.set_title("Every ALR phase on a time line: coloured by type when it has UVP images, red when it has none; "
                 "above, the gaps from the last UVP image to the next", loc="left", fontsize=10)
    fig.canvas.manager.set_window_title("QC – UVP coverage")
    path = out / "uvp_alr_coverage_map.png"
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
        gaps, short = coverage_gaps(match.phases, t_uvp)
        written["coverage_map"], _ = plot_coverage_map(out, match.phases, gaps)
        written["coverage_csv"] = write_coverage_csv(out, gaps)
        lines += coverage_table(match.phases, gaps, short)
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
