# %% 0. Setup

import sys
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

prompter = TkPrompter()
# ==========
# %% 1. Pick and load ALR data, set figure output directory
# ==========
ctd_dir = prompter.choose_directory(
    "Select ALR CTD data folder")
assert ctd_dir is not None, "cancelled"

nav_dir = prompter.choose_directory(
    "Select ALR NAV data folder")
assert nav_dir is not None, "cancelled"

if OUTPUT_DIR is None:
    out_dir = prompter.choose_directory("Select output folder for figures")
    assert out_dir is not None, "cancelled"
else:
    out_dir = Path(OUTPUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

ctd_files = sorted(alr_data.list_data_files(ctd_dir,"*.log"))
nav_files = sorted(alr_data.list_data_files(nav_dir,"*.csv"))
print(f"{len(ctd_files)} CTD log(s), {len(nav_files)} nav file(s)")

big_df = alr_data.load_all_files(ctd_files, 
                                 nav_files, 
                                 verbose=True)
big_df.write_parquet(REPO / "_cache_big_df.parquet")

# ==========
# %% 2. Profile Segmentation
# ==========
df, profiles_df = alr_data.detect_profiles(big_df, **PHASE_DETECTION)
df = alr_data.attach_profiles(df, profiles_df)

fig, _ = alr_plots.plot_phase_detection(profiles_df, **PHASE_DETECTION, show=False)
fig.savefig(Path(out_dir) / f"{ALR_SERIAL}_phase_detection.png", dpi=150)
plt.show()

fig, _ = alr_plots.plot_phase_sectioning(df, show=False)
fig.savefig(Path(out_dir) / f"{ALR_SERIAL}_phase_sectioning.png", dpi=150)
plt.show()

alr_plots.check_phases(df)  # prints the phase durations and any warning

# ==========
# %% 3. Build UVP project directories
# ==========
project = prompter.choose_directory("Select the UVP project folder")
assert project is not None, "cancelled"
project = Path(project)

meta_dir = project / "doc" / f"alr_meta_sn{ALR_SERIAL_NUMBER:03d}"
for d in ("raw", "meta", "CTDdata", "config"):
    (project / d).mkdir(parents=True, exist_ok=True)
meta_dir.mkdir(parents=True, exist_ok=True)

cruise_info = project / "config" / "cruise_info.txt"
if not cruise_info.exists():
    cruise_info.write_text(f"UVP project cruise info \n\
                           name: {CRUISE}\n",
                           encoding="latin-1")

written = alr_data.export_alr_netcdf(
    df,
    alr_serial=ALR_SERIAL,
    output_dir=str(meta_dir),
    max_files=None,
)

print(f"Written {len(written)} segment netCDF(s) to {meta_dir}")

# ==========
# %% 4. UVP processing
# ==========
sample_file = create_samples(project, 
                             cruise=CRUISE,
                             plots_dir=out_dir,
                             prompter=prompter,
                             assume_yes=False)
print(sample_file.read_text(encoding="cp1252"))
