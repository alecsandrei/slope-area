# slope-area

Tool that can be used to generate slope-area plots and other analysis.

## 📚 Table of Contents

1. [API Examples](#api-examples)
   - [Minimal example](#minimal-example)  
   - [Run trials with multiprocessing](#run-trials-with-multiprocessing)
   - [Create trials with Trial factories](#create-trials-with-trial-factories)
   - [Run trials with custom slope providers](#run-trials-with-custom-slope-providers)
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
context = TrialContext(
    out_dir=out_dir, data=TrialData(outlet, dem, resolution=None)
)
result = Trial(f'Trial {outlet.name}', context=context).run()
result.plot(
    config=SlopeAreaPlotConfig(hue=Column.SLOPE_TYPE, legend_font_size=8),
    out_fig=out_fig,
)
```

<img src="https://raw.githubusercontent.com/alecsandrei/slope-area/refs/heads/main/data/processed/00_minimal_example/slope_area.png" alt="drawing" width="400"/>

### Run trials with multiprocessing

```py
Trials([trial_1, trial_2, trial_3]).run(max_workers=3)
```

### Create trials with Trial factories

```py
resolutions = [(res, res) for res in range(30, 60, 5)]
context = TrialFactoryContext(
    dem=dem, out_dir=out_dir, analysis=analysis_config
)
trials = ResolutionTrialFactory(
    context=context, outlet=outlet, resolutions=resolutions
).generate()
results = trials.run(max_workers)
grid = results.plot(config=plot_config, out_fig=out_fig)
```

Console while running trials in parallel.

![console](https://raw.githubusercontent.com/alecsandrei/slope-area/refs/heads/main/assets/console.webp)

![slope-area-plot-2](https://raw.githubusercontent.com/alecsandrei/slope-area/refs/heads/main/data/processed/01_plot_from_outlets/resolution/slope_area.png)

### [Run trials with custom slope providers](https://github.com/alecsandrei/slope-area/blob/main/notebooks/03_custom_slope_providers.ipynb)

```py
slope_providers: SlopeProviders = {
    method_name: DefaultSlopeProviders.SAGASlope(method=i)
    for i, method_name in enumerate(
        (
            'maximum slope (Travis et al. 1975)',
            'maximum triangle slope (Tarboton 1997)',
            ...
        )
    )
}
...
analysis_config = AnalysisConfig(
    ...
    slope_providers=slope_providers,
)
```

![slope-area-plot-3](https://raw.githubusercontent.com/alecsandrei/slope-area/refs/heads/main/data/processed/03_custom_slope_providers/outlet/slope_area.png)

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
uv sync --frozen --all-groups
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
