# slope-area

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Tool that can be used to generate slope-area plots and other analysis.

## 📚 Table of Contents

1. [Installation](#installation)  
   - [Download SAGA GIS](#download-saga-gis)  
     - [Download SAGA GIS on Linux](#download-saga-gis-on-linux)  
     - [Download SAGA GIS on Windows](#download-saga-gis-on-windows)  
   - [Download package](#download-package)  
   - [Activate environment](#activate-environment)  
     - [Activate environment on Linux](#activate-environment-on-linux)  
     - [Activate environment on Windows](#activate-environment-on-windows)  
   - [Run tests](#run-tests)  
2. [Project Organization](#project-organization)  
3. [License](#license)

## Installation

### Download SAGA GIS

#### Download SAGA GIS on Linux

```sh
sudo apt install saga
```

#### Download SAGA GIS on Windows

Install the latest version on [SourceForge](https://sourceforge.net/projects/saga-gis/files/latest/download).

### Download package

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
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details.
├── logging
│   └── config.json    <- Python logging module configurations.
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description,
│                         e.g. `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         slope_area and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
└── slope_area   <- Source code for use in this project.
```

## License

MIT

--------
