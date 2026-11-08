"""Plotting of the outputs built by alr_data.py.

Nothing here loads, builds or writes data — every function takes a dataframe
that alr_data.py has already produced, and draws it.

    import alr_plots

    alr_plots.plot_pressure_overview(big_df)
    alr_plots.plot_phase_detection(profiles_df, gradient_thresholds=[0.05, -0.05], ...)
    alr_plots.plot_vertical_velocity(profiles_df)
    alr_plots.plot_series_pair(big_df_with_profiles, "Temperature", "position_lat")
    alr_plots.plot_phase_sectioning(big_df_with_profiles)

Each function returns (fig, axs) so the notebook can tweak the result further.

Phases are coloured as in the documentation (Table 4.1): downcast d, upcast u,
transect t, surface transect s and transition x.

Backend note: choose your matplotlib backend in the notebook, not here —
'%matplotlib inline' for figures in the notebook, or mpl.use('tkagg') for
interactive pan/zoom windows.
"""

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# Colour and name of each phase type, as in the documentation (Table 4.1, Figure 4.1)
PHASE_COLOURS = {"d": "#2a78d6", "u": "#eb6834", "t": "#1baf7a", "s": "#eda100", "x": "#9a9994"}
PHASE_NAMES = {"d": "Downcast", "u": "Upcast", "t": "Transect", "s": "Surface transect", "x": "Transition"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def truncate_cmap(name, minval=0.25, maxval=1.0, n=256):
    """Trim the pale end off a colormap so low values stay legible."""
    cmap = mpl.colormaps[name]
    return mpl.colors.LinearSegmentedColormap.from_list(
        f"trunc({name},{minval},{maxval})", cmap(np.linspace(minval, maxval, n))
    )


def to_pytime(series: pl.Series):
    """Convert Polars datetime column to numpy array of Python datetimes."""
    return np.array(series.to_list())


def split_by_flags(profiles_df):
    """Split a profiles dataframe into (profiles, not_profiles, transects, not_transects)."""
    clean = profiles_df.drop_nans()
    return (
        clean.filter(pl.col("is_profile").cast(pl.Boolean)),
        clean.filter(pl.col("is_profile").cast(pl.Boolean).not_()),
        clean.filter(pl.col("is_transect").cast(pl.Boolean)),
        clean.filter(pl.col("is_transect").cast(pl.Boolean).not_()),
    )


def to_datetime64(series: pl.Series) -> np.ndarray:
    """TIME column as datetime64[ns], whether it is still a Polars Datetime or
    find_profiles' NaN padding has turned it into float nanoseconds."""
    if series.dtype.is_temporal():
        return series.cast(pl.Datetime("ns")).to_numpy()
    ns = series.cast(pl.Float64).to_numpy()
    out = np.full(ns.shape, np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = np.isfinite(ns)
    out[ok] = ns[ok].astype("int64").astype("datetime64[ns]")
    return out


def phase_types(df, surface_thresh=5.0):
    """Phase letter (d, u, t, s or x) of every row, with the rules of
    alr_data.export_alr_netcdf: a profile is an upcast when its first depth is
    deeper than its last one, a transect is a surface transect when its mean
    depth is shallower than surface_thresh, and everything else is a transition.
    Records that find_profiles could not classify (NaN padding at the start and
    end) get "-": like the export, the figures leave them out."""
    depth = df["DEPTH"].cast(pl.Float64).to_numpy()
    types = np.full(depth.shape, "x")
    types[~np.isfinite(df["is_profile"].cast(pl.Float64).fill_null(np.nan).to_numpy())] = "-"
    for flag, num, kind in (("is_profile", "profile_num_new", "profile"),
                            ("is_transect", "transect_num", "transect")):
        member = df[flag].cast(pl.Float64).to_numpy() == 1
        ids = df[num].cast(pl.Float64).to_numpy()
        for i in np.unique(ids[member]):
            rows = np.flatnonzero(member & (ids == i))
            z = depth[rows]
            z = z[np.isfinite(z)]
            if kind == "profile":
                types[rows] = "u" if z.size and z[0] > z[-1] else "d"
            else:
                types[rows] = "s" if z.size and z.mean() < surface_thresh else "t"
    return types


def phase_runs(df, surface_thresh=5.0):
    """The phases as runs of rows, as alr_data.export_alr_netcdf writes them:
    a new run when the phase type or the profile/transect number changes.

    Returns (t, z, types, runs, seg_starts): runs = [(first row, end row, letter)],
    seg_starts = start times of the downcasts beginning shallower than surface_thresh."""
    t = to_datetime64(df["TIME"])
    z = df["DEPTH"].cast(pl.Float64).to_numpy()
    types = phase_types(df, surface_thresh)
    pid = df["profile_num_new"].cast(pl.Float64).fill_nan(-1).fill_null(-1).to_numpy()
    tid = df["transect_num"].cast(pl.Float64).fill_nan(-1).fill_null(-1).to_numpy()
    group = np.where(np.isin(types, ["d", "u"]), pid, np.where(np.isin(types, ["t", "s"]), tid, -1))
    change = np.flatnonzero((types[1:] != types[:-1]) | (group[1:] != group[:-1])) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, types.size]
    runs = [(s, e, types[s]) for s, e in zip(starts, ends) if types[s] != "-"]
    seg_starts = [t[s] for s, e, k in runs if k == "d" and np.isfinite(z[s]) and z[s] < surface_thresh]
    return t, z, types, runs, seg_starts


def check_phases(big_df_with_profiles, surface_thresh=5.0, max_transition_min=30.0, verbose=True):
    """Consistency checks of the sectioning; returns the warnings (also printed).

    - every segment after the first (which holds the pre-dive) has an upcast;
    - no transition is longer than max_transition_min (a missed profile or transect);
    - the duration of each phase type, to spot thresholds that suit the deployment badly.
    """
    t, z, types, runs, seg_starts = phase_runs(big_df_with_profiles, surface_thresh)
    minutes = lambda s, e: float((t[e - 1] - t[s]) / np.timedelta64(1, "m"))
    warnings_out = []

    bounds = list(seg_starts) + [t[runs[-1][1] - 1] + np.timedelta64(1, "s")] if runs else []
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]), start=1):
        letters = {k for s, e, k in runs if a <= t[s] < b}
        if "u" not in letters:
            warnings_out.append(f"segment {i:04d} (starting {np.datetime_as_string(a, unit='m')}) has no upcast")

    first_dive = seg_starts[0] if seg_starts else None
    for s, e, k in runs:
        if first_dive is not None and t[s] < first_dive:
            continue  # segment 0000: the pre-dive, at the surface before the first downcast
        if k == "x" and minutes(s, e) > max_transition_min:
            warnings_out.append(f"transition of {minutes(s, e):.0f} min starting {np.datetime_as_string(t[s], unit='m')}"
                                f" (> {max_transition_min:g} min: a missed profile or transect?)")

    if verbose:
        print(f"--- Phase checks: {len(runs)} phases, {len(seg_starts) + 1} segments ---")
        for k in PHASE_COLOURS:
            d = [minutes(s, e) for s, e, kk in runs if kk == k]
            if d:
                print(f"  {k} – {PHASE_NAMES[k]:<16} {len(d):4d} phases, duration median {np.median(d):6.1f} min,"
                      f" range {min(d):.1f}–{max(d):.1f} min")
        for w in warnings_out:
            print(f"  WARNING: {w}")
        if not warnings_out:
            print("  no warning")
    return warnings_out


def _scatter_by_phase(ax, t, y, types, ms=1.5):
    for k, colour in PHASE_COLOURS.items():
        m = types == k
        if m.any():
            ax.plot(t[m], y[m], ".", ms=ms, ls="", color=colour, rasterized=True)


def _phase_legend(fig, types):
    handles = [plt.Line2D([], [], color=PHASE_COLOURS[k], lw=4, label=f"{k} – {PHASE_NAMES[k]}")
               for k in PHASE_COLOURS if (types == k).any()]
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), frameon=False)


def _format_time_axis(ax):
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _window_title(fig, title):
    """Name the interactive window (otherwise 'Figure 1', 'Figure 2', ...)."""
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(title)


# ---------------------------------------------------------------------------
# 2.4  Inspecting the data
# ---------------------------------------------------------------------------

def plot_pressure_overview(big_df, font_size=18, show=True):
    """Pressure against time for the whole built dataframe (pre-profiling)."""
    plt.rcParams.update({
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
    })

    # Convert TIME column
    df_plot = big_df.with_columns(
        (pl.col("Timestamp") * 1e9).cast(pl.Datetime("ns")).alias("TIME")
    )

    time_vals = df_plot["TIME"].to_numpy()
    pressure_vals = df_plot["Pressure"].to_numpy()

    fig, axs = plt.subplots(1, 1, figsize=(18, 8), sharex=True)

    axs.plot(time_vals, pressure_vals, marker=".", markersize=1, ls="", c="tab:blue")
    axs.set_ylabel("Pressure (dbar)")
    axs.invert_yaxis()

    plt.tight_layout()
    if show:
        plt.show()

    return fig, axs


# ---------------------------------------------------------------------------
# 3.1  Phase detection: the criteria behind each decision
# ---------------------------------------------------------------------------

def plot_phase_detection(
    profiles_df,
    gradient_thresholds=[0.05, -0.05],
    cust_gradient_thresholds=[0.005, -0.005],
    custom_column="pitch",
    surface_thresh=5.0,
    strict_profiles=False,
    use_only_pit_vel=False,
    show=True,
    **_other_detection_params,
):
    """Five panels sharing the time axis, to check why each record got its phase.

    Panels: depth; criterion 1 (smoothed vertical velocity); pitch; criterion 2
    (M = -pitch x vertical velocity, Eq. 2 of the documentation); profile
    numbering before and after the duration and back-fill filters.

    Pass the same thresholds as to alr_data.detect_profiles (any other detection
    parameter is accepted and ignored), so that the figure shows the thresholds
    actually used.
    """
    t = to_datetime64(profiles_df["TIME"])
    types = phase_types(profiles_df, surface_thresh)

    def col(name):
        return profiles_df[name].cast(pl.Float64).to_numpy()

    pos, neg = gradient_thresholds
    cpos, cneg = cust_gradient_thresholds

    fig, axs = plt.subplots(5, 1, figsize=(16, 12), height_ratios=[3, 3, 2.2, 2.2, 1.4],
                            sharex=True, constrained_layout=True)

    ax = axs[0]
    _scatter_by_phase(ax, t, col("INTERP_DEPTH"), types)
    ax.invert_yaxis()
    ax.set_ylabel("Depth (dbar)")
    ax.set_title("Phases detected", loc="left")

    ax = axs[1]
    w_raw, w = col("grad"), col("smooth_grad")
    ax.axhspan(neg, pos, color="0.92", zorder=0)
    ax.plot(t, w_raw, color="0.75", lw=0.4, zorder=1, rasterized=True)
    _scatter_by_phase(ax, t, w, types)
    lo, hi = np.nanpercentile(w_raw, [0.5, 99.5])
    ax.set_ylim(min(lo, 2 * neg), max(hi, 2 * pos))
    ax.set_ylabel("Vertical velocity\n(dbar s⁻¹)")
    ax.set_title(f"Criterion 1 – smoothed vertical velocity w̃ (raw in grey): "
                 f"profile when w̃ is outside the grey band [{neg:g}, {pos:g}]", loc="left")

    smooth_name = f"smooth_INTERP_{custom_column}"
    if smooth_name in profiles_df.columns:
        theta = col(smooth_name)
        ax = axs[2]
        if custom_column in profiles_df.columns:
            ax.plot(t, col(custom_column), color="0.75", lw=0.4, zorder=1, rasterized=True)
        _scatter_by_phase(ax, t, theta, types)
        ax.set_ylabel(f"{custom_column.capitalize()}")
        ax.set_title(f"Smoothed {custom_column} θ̃ (raw in grey)", loc="left")

        ax = axs[3]
        m = -theta * w
        ax.axhspan(cneg, cpos, color="0.92", zorder=0)
        _scatter_by_phase(ax, t, m, types)
        lo, hi = np.nanpercentile(m, [0.5, 99.5])
        ax.set_ylim(min(lo, 2 * cneg), max(hi, 4 * cpos))
        ax.set_ylabel("M = −θ̃ · w̃")
        ax.set_title(f"Criterion 2 – product M (Eq. 2): profile when M is outside the grey band "
                     f"[{cneg:g}, {cpos:g}]; a transect also needs M < {cpos:g}", loc="left")
    else:
        for ax in axs[2:4]:
            ax.text(0.5, 0.5, f"no '{custom_column}' column", transform=ax.transAxes, ha="center")

    ax = axs[4]
    for name, colour, label in (("profile_num", "0.6", "before filters"),
                                ("profile_num_new", "#1f1f1e", "after minimum-duration and back-fill filters")):
        n = col(name)
        n[n < 0] = np.nan  # -1 means "not a profile"
        ax.plot(t, n, color=colour, lw=1, label=label)
    ax.set_ylabel("Profile no.")
    ax.set_title("Profile numbering", loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    _format_time_axis(ax)

    if use_only_pit_vel:
        rule = "a record is part of a profile when criterion 2 is met"
    else:
        rule = f"a record is part of a profile when criterion 1 {'and' if strict_profiles else 'or'} 2 is met"
    fig.suptitle(f"ALR phase detection – {rule}", x=0.01, ha="left", fontweight="bold")
    _phase_legend(fig, types)
    _window_title(fig, "ALR phase detection – criteria")

    if show:
        plt.show()

    return fig, axs


# ---------------------------------------------------------------------------
# 3.2  Plotting just vertical velocities
# ---------------------------------------------------------------------------

def plot_vertical_velocity(profiles_df, ylim=(-0.5, 0.5), font_size=16,
                           font_family="Arial", show=True):
    """Vertical velocity alone, on a fixed y-axis, coloured by classification."""
    plt.rcParams.update({
        "font.family": font_family,
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
    })

    profiles, not_profiles, transects, _ = split_by_flags(profiles_df)

    fig, axs = plt.subplots(1, 1, figsize=(18, 3), sharex=True)

    axs.set_xlabel("Time")
    axs.set_ylabel("Vertical Velocity (m/s)")
    axs.set_ylim(list(ylim))

    for data, col, label in zip(
            [profiles, not_profiles, transects],
            ["tab:blue", "tab:red", "tab:orange"],
            ["Profile", "Not Profile", "Transect"]):
        axs.plot(
            data["TIME"].to_numpy(),
            data["smooth_grad"].to_numpy(),
            marker=".", markersize=1, ls="", c=col, label=label
        )

    axs.legend(loc="upper right")

    plt.tight_layout()
    if show:
        plt.show()

    return fig, axs


# ---------------------------------------------------------------------------
# 4.0  Annotated dataframe
# ---------------------------------------------------------------------------

def plot_series_pair(big_df_with_profiles, var1="Temperature", var2="position_lat", show=True):
    """Two stacked panels of any two columns against time, sharing the x-axis."""
    df_plot = big_df_with_profiles

    time_vals = df_plot["TIME"].to_numpy()
    var1_vals = df_plot[var1].to_numpy()
    var2_vals = df_plot[var2].to_numpy()

    fig, axs = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

    axs[0].plot(time_vals, var1_vals, marker=".", markersize=1, ls="", c="tab:blue")
    axs[0].set(ylabel=var1)
    axs[0].invert_yaxis()

    axs[1].plot(time_vals, var2_vals, marker=".", markersize=1, ls="", c="purple")
    axs[1].set(xlabel="Time", ylabel=var2)

    plt.tight_layout()
    if show:
        plt.show()

    return fig, axs


def plot_phase_sectioning(big_df_with_profiles, surface_thresh=5.0, show=True):
    """Result of the sectioning: the deployment coloured by phase, and below it
    one block per phase (one block = one netCDF file = one sample), with the
    segment boundaries (a new segment at each downcast starting shallower than
    surface_thresh)."""
    t, z, types, runs, seg_starts = phase_runs(big_df_with_profiles, surface_thresh)
    counts = {k: sum(1 for *_, kk in runs if kk == k) for k in PHASE_COLOURS}

    fig, (ax, ax_s) = plt.subplots(2, 1, figsize=(16, 8), height_ratios=[5, 1],
                                   sharex=True, constrained_layout=True)
    for s, e, k in runs:
        ax.plot(t[s:e], z[s:e], color=PHASE_COLOURS[k], lw=1.4)
    ax.invert_yaxis()
    ax.set_ylabel("Depth (dbar)")
    ax.axhline(surface_thresh, color="0.6", ls="--", lw=0.8)
    ax.text(0.995, surface_thresh, f"surface threshold {surface_thresh:g} dbar", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, color="0.4")

    x = mdates.date2num(t)
    for k, colour in PHASE_COLOURS.items():
        bars = [(x[s], x[e - 1] - x[s]) for s, e, kk in runs if kk == k]
        if bars:
            ax_s.broken_barh(bars, (0.1, 0.8), facecolors=colour, edgecolors="white", linewidth=0.5)
    ax_s.set_ylim(0, 1)
    ax_s.set_yticks([])
    ax_s.set_ylabel("Phases")
    for a in (ax, ax_s):
        for s in seg_starts:
            a.axvline(s, color="0.5", ls=":", lw=0.8)
    _format_time_axis(ax_s)

    summary = (f"{counts['d']} downcasts, {counts['u']} upcasts, {counts['t']} transects, "
               f"{counts['s']} surface transects, {counts['x']} transitions – {len(seg_starts) + 1} segments "
               f"(dotted lines)")
    ax.set_title(f"{len(runs)} phases: {summary}", loc="left")
    fig.suptitle("ALR phase sectioning – result (one phase = one netCDF file = one sample)",
                 x=0.01, ha="left", fontweight="bold")
    _phase_legend(fig, types)
    _window_title(fig, "ALR phase sectioning – result")

    if show:
        plt.show()

    return fig, (ax, ax_s)


# Former names, kept so that existing notebooks keep working
plot_profile_detection = plot_phase_detection
plot_profiles_and_transects = plot_phase_sectioning
