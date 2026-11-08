# %% 0. Setup

import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO / "main"), str(REPO / "UVP_toolbox_Python")]

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import alr_data
import alr_plots
from uvp_toolbox.create_samples import create_samples
from uvp_toolbox.prompts import TkPrompter

ALR_SERIAL_NUMBER = 4
ALR_SERIAL = f"ALR{ALR_SERIAL_NUMBER:03d}"
CRUISE = "DY180"
OUTPUT_DIR = None # Output for figures; set here to skip prompts asking for figure directory
# Phase detection thresholds, used by the detection and shown on its figure;
# any other alr_data.detect_profiles parameter can be added here
PHASE_DETECTION = dict(gradient_thresholds=[0.05, -0.05],
                       cust_gradient_thresholds=[0.005, -0.005])
# Sample file 'integrationtime': depth bin (m) of the depth-integrated samples
# (d, u, x) and time bin (s) of the time-integrated samples (t, s)
DEPTH_INTEGRATION = 5
TIME_INTEGRATION = 3600
# Folders of a UVPapp project, checked in cell 3: the pipeline does not create them
UVP_PROJECT_DIRS = ("calibration", "config", "CTDdata", "doc", "ecodata",
                    "livecamera", "log", "meta", "raw", "results")

prompter = TkPrompter()
# ==========
# %% 1. Load and parse the ALR data (CTD and NAV)
# ==========
ctd_dir = prompter.choose_directory(
    "Select the ALR CTD files folder")
assert ctd_dir is not None, "cancelled"

nav_dir = prompter.choose_directory(
    "Select the ALR NAV files folder")
assert nav_dir is not None, "cancelled"

if OUTPUT_DIR is None:
    out_dir = prompter.choose_directory("Select the figures folder")
    assert out_dir is not None, "cancelled"
else:
    out_dir = OUTPUT_DIR
out_dir = Path(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

ctd_files = sorted(alr_data.list_data_files(ctd_dir,"*.log"))
nav_files = sorted(alr_data.list_data_files(nav_dir,"*.csv"))
print(f"{len(ctd_files)} CTD log(s), {len(nav_files)} nav file(s)")

big_df = alr_data.load_all_files(ctd_files,
                                 nav_files,
                                 verbose=True)
big_df.write_parquet(REPO / "_cache_big_df.parquet")

# ==========
# %% 2. Identify the ALR phases (profiles, transects, transitions)
# ==========
df, profiles_df = alr_data.detect_profiles(big_df, **PHASE_DETECTION)
df = alr_data.attach_profiles(df, profiles_df)

fig, _ = alr_plots.plot_phase_detection(profiles_df, **PHASE_DETECTION, show=False)
fig.savefig(out_dir / f"{ALR_SERIAL}_phase_detection.png", dpi=150)
plt.show()

fig, _ = alr_plots.plot_phase_sectioning(df, show=False)
fig.savefig(out_dir / f"{ALR_SERIAL}_phase_sectioning.png", dpi=150)
plt.show()

alr_plots.check_phases(df)  # prints the phase durations and any warning

# ==========
# %% 3. Generate one netCDF file per ALR phase
# ==========
project = prompter.choose_directory("Select the UVP6 project folder")
assert project is not None, "cancelled"
project = Path(project)

# A complete UVPapp project is expected (as ALR4_demo): its folders are not created here
missing = [d for d in UVP_PROJECT_DIRS if not (project / d).is_dir()]
assert not missing, f"Not a complete UVPapp project, missing folder(s): {', '.join(missing)}"

# The ALR netCDFs go to meta/, next to the sample file: raw, config and meta
# are the folders kept and backed up for a UVP project
meta_dir = project / "meta" / f"alr_meta_sn{ALR_SERIAL_NUMBER:03d}"
meta_dir.mkdir(exist_ok=True)

cruise_info = project / "config" / "cruise_info.txt"
if not cruise_info.exists():
    cruise_info.write_text(f"UVP project cruise info \n\
                           name: {CRUISE}\n",
                           encoding="latin-1")

# netCDFs of a previous run are moved to archives/, as cell 4 reads every .nc of meta_dir
old_nc = sorted(meta_dir.glob("*.nc"))
if old_nc:
    archive = meta_dir / "archives" / datetime.now().strftime("%Y%m%d-%H%M%S")
    archive.mkdir(parents=True)
    for f in old_nc:
        shutil.move(str(f), str(archive / f.name))
    print(f"Moved {len(old_nc)} netCDF(s) of the previous run to {archive}")

written = alr_data.export_alr_netcdf(
    df,
    alr_serial=ALR_SERIAL,
    output_dir=str(meta_dir),
    max_files=None,
)

print(f"Written {len(written)} phase netCDF(s) to {meta_dir}")

# ==========
# %% 4. Generate the UVPapp sample file (header file)
# ==========
sample_file = create_samples(project,
                             cruise=CRUISE,
                             plots_dir=out_dir,
                             depth_integration=DEPTH_INTEGRATION,
                             time_integration=TIME_INTEGRATION,
                             prompter=prompter,
                             assume_yes=False)
print(sample_file.read_text(encoding="cp1252"))
