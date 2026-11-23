<img src="assets/auv_app.png" alt="AUV-UVP6 app icon" width="110" align="right">

# UVP6 AUV sample creation

Creates the UVPapp sample file of a UVP6 deployed on an autonomous underwater vehicle (AUV), from
the AUV data: the deployment is cut into phases (downcasts, upcasts, transects, surface
transects, transitions), and each phase becomes one sample. It was developed and tested with the
Autosub Long Range (ALR), whose CTD and navigation files it reads directly; other AUVs give their
data as simple CSV files.

- **Authors**: Hans Hilder ([ORCID 0009-0008-8882-0754](https://orcid.org/0009-0008-8882-0754)) and Amanda Elineau
  ([ORCID 0000-0001-6724-9095](https://orcid.org/0000-0001-6724-9095)), National Oceanography Centre, Southampton, UK.
  To cite the code, use **Cite this repository** on this page ([`CITATION.cff`](CITATION.cff)).
- **Protocol**: to be published in the [Ocean Best Practices System](https://repository.oceanbestpractices.org) (DOI to come).
- **Demo data**: [ALR4_demo.zip](https://github.com/hans-hilder/uvp6-auv-sample-creation/releases/tag/v1.0_demodataset)
  (231 MB): ALR CTD and NAV files, a complete UVP6 project, and the outputs of a run (sample
  file, phase netCDFs, CTD files and figures). Running it again archives these outputs.
- **Details** (settings, outputs, quality-control figures): [docs/details.md](docs/details.md).

## Install (once, Windows)

1. **Code**: green **Code** button at the top of this page → **Download ZIP**, and extract it,
   e.g. to `C:\AUV\uvp6-auv-sample-creation`.
2. **Demo data** (optional): extract `ALR4_demo.zip` to a short path, e.g. `C:\AUV\ALR4_demo`.
   Windows cannot write files whose path is longer than 260 characters (the app warns you).
3. **Miniconda** (free, it installs Python): [anaconda.com/download](https://www.anaconda.com/download/success),
   default options ("Just Me", keep **Create shortcuts**).
4. **Environment**: open **Anaconda Prompt** from the Start menu and type:
   ```
   cd C:\AUV\uvp6-auv-sample-creation
   conda env create -f environment.yml
   ```
5. **Desktop shortcut** (optional): double-click [`create_shortcut.bat`](create_shortcut.bat).

## Run with the app

Double-click the **AUV-UVP6 app** shortcut, or [`start_app.bat`](start_app.bat). A black window
opens (keep it open while you use the app), then the app opens in your web browser at
http://localhost:8501. It runs on your computer, on your data: nothing is sent anywhere.

1. **Data**: click **Browse…** and choose the deployment folder, e.g. `C:\AUV\ALR4_demo`. Its
   folders are found and checked: each title must turn green.
2. **Settings**: check the AUV name and serial number, the cruise name and ID, the ship and the sample
   integration.
3. **Run**: click **Run the pipeline**.
4. **Results**: download the sample file, and look at the checks and the figures.

To share the app, share this repository: each user installs it and runs it on their own computer.
If the black window shows "python is not recognized", type in Anaconda Prompt:
`conda activate auv_uvp`, `cd C:\AUV\uvp6-auv-sample-creation`, then `python -m streamlit run app.py`.

## Run with the script

Open [`run_UVP_toolbox.py`](run_UVP_toolbox.py) in VS Code (Python and Jupyter extensions,
`auv_uvp` interpreter), set `VEHICLE_NAMESN`, `AUV_DATA_FORMAT`, `CRUISE_NAMEID`, `SHIP_NAME` and the other settings at
the top, and click **Run Cell** above each `# %%`, in order:

| Cell | Asks for | What happens |
| --- | --- | --- |
| 1. Load and parse the AUV data | ALR CTD and NAV folders (or the CSV file), figures folder | Reads and merges the AUV data files. |
| 2. Identify the AUV phases | – | Detects the phases, saves two figures, prints the phase checks. |
| 3. Generate one netCDF file per AUV phase | UVP6 project folder | Checks the project folders, writes one netCDF per phase. |
| 4. Generate the UVPapp sample file | Yes/No if a sample file exists | Matches the UVP6 images to the phases, writes the sample file, the CTD files and the figures. |

Close UVPapp before running cell 4.

## Data and outputs

```
<deployment folder>/                         e.g. ALR4_demo
├── ALR004_ctd_data/*.log                    ALR CTD files
├── ALR004_nav_data/*.csv                    ALR navigation files
│   (other AUVs: one <AUV name>.csv file     simple CSV file, instead of the two ALR folders)
├── diagnostic_figures/                      figures and quality checks (written)
└── uvp6_<sn>_<date>_<cruise>_<platform>/    complete UVPapp project, sequences merged in UVPapp
    ├── raw/<yyyymmdd-HHMMSS>/               UVP6 sequence (read)
    ├── meta/auv_meta_alr004/*.nc            one netCDF per AUV phase (written)
    ├── meta/uvp6_header_<project>.txt       UVPapp sample file (written)
    └── CTDdata/*.ctd                        one CTD file per sample (written)
```

The simple CSV files have the columns `time`, `pressure` (or `depth`), `pitch`, `latitude`,
`longitude` (see [docs/details.md](docs/details.md#data)).

Files of a previous run are moved to `archives/` folders, not overwritten.
