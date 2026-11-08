# ALR UVP6 Processing


Scripts for processing UVP6 data from the ALR platform.

[`run_UVP_toolbox.py`](run_UVP_toolbox.py) runs the processing:

1) Reading in ALR CTD and navigation logs

2) Profile/transect detection from CTD/NAV data

3) One netCDF created per ALR segment

4) UVPApp sample file creation

This script is tested only for use with ALR generated data.

## Setup

One Python environment is necessary, with the packages in [`requirements.txt`](requirements.txt), e.g.

```bash
conda activate <env>
pip install -r requirements.txt
```

VSCode is recommended for running this script.
Select that environment as the Python interpreter in VS Code.

## Before you run

### 1. **Close UVPApp**

### 2. **ALR data prep** 
  Have one folder of CTD `.log` files and one folder of navigation
  `.csv` files. Every file directly inside each folder is read and merged, subfolders are ignored. 
  <br>
  #TODO In future releases, this step will be
  simplified by a combined .nc (netCDF) file for ALR data
  </br>

  ```
  
  <parent_directory>/
  └──<alr_serial_number>_ctd_data/
      └──*.log
  └──<alr_serial_number>_nav_data/
      └──*.csv
  
  ```

### 3. **UVP project folder prep**
With the UVP6 sequences in `raw/`:

  ```

  <parent_directory>/
  └──uvp6_<uvp6_serial_number>_<project_creation_data>_<cruise>_<project>_<platform><platform_serial_number>/
      └── raw/<yyyymmdd-HHMMSS>/<yyyymmdd-HHMMSS>_data.txt
  
  ```
  We recommended creating a folder for figure outputs, e.g.:
  ```

  <parent_directory>/
  └──diagnostic_figures/

  ```

### 4. **Processing settings** 
At the top of the [`run_UVP_toolbox.py`](run_UVP_toolbox.py):

  | Setting | What it does |
  | --- | --- |
  | `ALR_SERIAL_NUMBER` | ALR serial, e.g. `4` becomes `ALR004`. Names the netCDF files and the `doc/alr_meta_sn004` folder. |
  | `CRUISE` | Cruise name/project name associated with deployment, written to the sample file. |
  | `OUTPUT_DIR` | Folder for all figures. `None` asks with a dialog. |
  | `PHASE_DETECTION` | Thresholds of the phase detection (any other `alr_data.detect_profiles` parameter can be added). The same values are drawn on the phase detection figure. |

  In cell 4, `assume_yes=False` asks before replacing an existing sample file;
  `True` replaces it without asking.

## Run

Open `run_UVP_toolbox.py` in VS Code and click **Run Cell** above each `# %%`
marker, in order; the cells run in an Interactive Window. To run it all at
once instead: `python run_UVP_toolbox.py` in terminal.

| Cell | Asks for (in order) | What happens |
| --- | --- | --- |
| 0. Setup | – | Imports and settings. |
| 1. Load ALR data | CTD folder, NAV folder, figures folder | Reads and merges CTD and NAV files. |
| 2. Profile segmentation | – | Detects profiles and transects; displays and saves two figures (see [Quality-control figures](#quality-control-figures)). |
| 3. Build UVP project | UVP project folder | Creates the project subfolders; writes one netCDF per ALR segment. |
| 4. UVP processing | Yes/No if a sample file exists) | Matches UVP sequences to ALR segments; writes the sample file, CTD files and diagnostic figures. |

Figure windows open one at a time: close each one to continue.

Previous sample files are NOT deleted, they are renamed with a '_YYYYMMDD-HHMMSS' suffix.

Near the end of cell 4, two lines summarise CTD/NAV file and UVP image matching:

- `Matching UVP data times were not found for X of Y CTD/NAV files`: ALR
  segments with no UVP data in their time window.
- `Matching CTD/NAV files were not found for X of Y UVP images`: UVP images
  outside every matched segment.

## Outputs

```
<UVP project>/
├── config/cruise_info.txt              # created if missing
├── doc/alr_meta_sn004/*.nc             # one per ALR segment (cell 3)
├── meta/uvp6_header_<project>.txt      # UVPApp sample file (cell 4)
└── CTDdata/<segment>.ctd               # one per matched segment (cell 4)

<figures folder>/
├── ALR004_phase_detection.png          # cell 2
├── ALR004_phase_sectioning.png         # cell 2
├── uvp_alr_timedepth.png               # cell 4, and below
├── uvp_alr_timedepth_fullspan.png
├── uvp_alr_sampling_intervals.png
├── daily_plots/uvp_alr_timedepth_<date>.png
├── overview_plots/uvp_alr_timedepth_all_days.png
├── uvp_alr_depth_agreement.png         # cell 4, quality checks
├── uvp_alr_phase_coverage.png
├── uvp_alr_sample_sizes.png
└── qc_summary.txt
```

Cell 1 saves the merged ALR data to `_cache_big_df.parquet` within the repository directory .
<br>#TODO Give `_cache_big_df.parquet` a more sensible name.

### Quality-control figures

Phases are coloured as in the documentation: d downcast, u upcast, t transect,
s surface transect, x transition.

| Figure | What it shows | What to check |
| --- | --- | --- |
| `ALR004_phase_detection.png` | Depth, the two detection criteria (smoothed vertical velocity, and M = −pitch × vertical velocity) with their thresholds as grey bands, the pitch, and the profile numbering before/after the duration and back-fill filters. | Why a record got its phase; whether the thresholds suit the deployment. |
| `ALR004_phase_sectioning.png` | The deployment coloured by phase, the count of each phase type, and one block per phase (one phase = one netCDF file = one sample), with the segment boundaries. | Each dive split into a downcast and an upcast, no missed profile, no fragmented transect. |
| `uvp_alr_timedepth.png`, `…_fullspan.png`, `daily_plots/`, `overview_plots/` | UVP (dots) and ALR (line) depth, the UVP − ALR depth difference (10 s medians), and, in `…_fullspan.png`, the ALR → UVP clock offset of each 31 × 10 s window, coloured by r²; the median over reliable windows (r² ≥ 0.9) is printed and drawn. | A constant depth difference points to a pressure offset, a difference growing with depth to a scale difference, and a non-zero median offset to a clock offset. |
| `uvp_alr_sampling_intervals.png` | Distribution of the time between two UVP images and two ALR records. | Gaps or irregular sampling. |
| `uvp_alr_depth_agreement.png` | UVP − ALR depth against ALR depth (10 s medians), with the fit UVP − ALR = scale × depth + offset + lag × vertical velocity, over the deployment and per day. | A WARNING when \|scale\| > 0.5 %, \|offset\| > 0.5 m or \|lag\| > 1 s. A scale near +1.97 % is what a UVP depth in mH₂O against an ALR pressure in dbar gives. A lag that grows day after day is a drifting clock. |
| `uvp_alr_phase_coverage.png` | Every ALR phase on a time line, by type: coloured when it has UVP images, grey with a red outline when it has none. | Periods without UVP data; phases that should have had images. |
| `uvp_alr_sample_sizes.png` | Images per sample, by phase type, with the `qc_min_images` limit of `create_samples` (20 by default). | Very small samples, usually transitions, not worth processing in UVPapp. |

The quality checks of cell 4 are also printed and written to `qc_summary.txt`. Cell 2 prints the
phase checks of `alr_plots.check_phases`: the duration of each phase type, a warning for a segment
without an upcast, and a warning for a transition longer than 30 min.

## Re-running

- **After editing code in `main/` or `UVP_toolbox_Python/`**, restart the
  interactive window's kernel.
- **Before re-running cell 3 with different profile detection settings**, empty
  `doc/alr_meta_sn004/`. Old netCDFs aren't removed, and cell 4 matches every
  `.nc` file in that folder.
  <br> #TODO Save a `config` file for profile detection.
  <br> #TODO Automatically move old netCDF files on rerun.
- To reiterate, previous sample files are NOT deleted, they are renamed with a '_YYYYMMDD-HHMMSS' suffix, e.g.
  `uvp6_header_<project>_<YYYYMMDD-HHMMSS>.txt`, and kept within the same /meta folder
