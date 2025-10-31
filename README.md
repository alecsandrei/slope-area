# slope-area

Tool that can be used to generate slope-area plots and other analysis.

## 📚 Table of Contents

1. [API Examples](#api-examples)
   - [Minimal example](#minimal-example)  
   - [Run trials with multiprocessing](#run-trials-with-multiprocessing)
   - [Run trials with Builder objects](#run-trials-with-builder-objects)  
2. [Installation](#installation)  
   - [Download SAGA GIS](#download-saga-gis)  
     - [Download SAGA GIS on Linux](#download-saga-gis-on-linux)  
     - [Download SAGA GIS on Windows](#download-saga-gis-on-windows)  
   - [Download uv](#download-uv)
   - [Download slope-area](#download-slope-area)  
   - [Activate environment](#activate-environment)  
     - [Activate environment on Linux](#activate-environment-on-linux)  
     - [Activate environment on Windows](#activate-environment-on-windows)  
   - [Run tests](#run-tests)  
3. [Project Organization](#project-organization)  
4. [License](#license)

## API examples

[Detailed notebooks here](https://github.com/alecsandrei/slope-area/tree/main/notebooks)

### Minimal example

```py
outlet = Outlet.from_xy(711339, 533362, name='outlet')
trial_config = TrialConfig(
    outlet.name,
    outlet,
    dem=dem,
    hydrologic_analysis_config=HydrologicAnalysisConfig(
        streams_flow_accumulation_threshold=1000, outlet_snap_distance=100
    ),
    out_dir=out_dir,
)
trial = Trial(trial_config).run()
slope_area_plot(
    data=trial.profiles,
    out_fig=out_fig,
    config=SlopeAreaPlotConfig(hue='slope_type'),
)
```
<img src="https://raw.githubusercontent.com/alecsandrei/slope-area/refs/heads/main/data/processed/00_minimal_example/slope_area.png" alt="drawing" width="600"/>

### Run trials with multiprocessing

```py
Trials([trial_1, trial_2, trial_3]).run(max_workers=3)
```

### Run trials with Builder objects

```py
resolutions = [(res, res) for res in range(5, 15)]
builder_config = BuilderConfig(
    hydrologic_analysis_config=HydrologicAnalysisConfig(
        streams_flow_accumulation_threshold=1000, outlet_snap_distance=100
    ),
    out_dir=out_dir,
    out_fig=out_fig,
    plot_config=SlopeAreaPlotConfig(hue='slope_type'),
    max_workers=max_workers,
)
results = ResolutionPlotBuilder(
    builder_config, dem, outlet, resolutions
).build()
```

![console](https://raw.githubusercontent.com/alecsandrei/slope-area/refs/heads/main/assets/console.webp)

![slope-area-plot-2](https://raw.githubusercontent.com/alecsandrei/slope-area/refs/heads/main/data/processed/02_internal_example/resolution_builder/slope_area.png)

## Installation

### Download SAGA GIS

Tip: define the saga_cmd environment variable to point to the saga_cmd file.
This may not be needed on Linux.

#### Download SAGA GIS on Linux

```sh
sudo apt install saga
```

#### Download SAGA GIS on Windows

Install the latest version on [SourceForge](https://sourceforge.net/projects/saga-gis/files/latest/download).

### Download uv

Details: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Download slope-area

```sh
git clone https://github.com/alecsandrei/slope-area
cd slope-area
uv sync --frozen --dev
```

### Activate environment

#### Activate environment on Linux

```sh
source .venv/bin/activate
```

#### Activate environment on Windows

```sh
.venv\Scripts\activate
```

### Run tests

```sh
pytest tests
```

## Project Organization

```text
├── LICENSE            <- Open-source license.
├── README.md          <- The top-level README.
├── data
│   │── raw            <- The original, immutable data dump.
│   └── processed      <- Output files from computation, used in notebook examples.
├── logging
│   └── config.json    <- Python logging module configurations.
├── notebooks          <- Jupyter notebooks. Naming convention is `01_workflow_name.ipynb`.
│
├── pyproject.toml     <- Python project configuration file.
└── slope_area         <- Source code for use in this project.
```

## License

MIT

--------
