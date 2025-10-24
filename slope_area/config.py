from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
LOGGING_CONFIG = PROJ_ROOT / 'logging' / 'config.json'
DATA_DIR = PROJ_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
INTERIM_DATA_DIR = DATA_DIR / 'interim'

load_dotenv(PROJ_ROOT / '.env', verbose=True)

DEM_DIR = Path(os.environ['DEM_DIR'])
DEM_DIR_EPSG = int(os.environ['DEM_DIR_EPSG'])
GENERALIZED_DEM = Path(os.environ['GENERALIZED_DEM'])
SAGA_CMD = os.environ.get('SAGA_CMD', None)
OUTLET_SNAP_DIST = float(os.environ.get('OUTLET_SNAP_DIST', 100))
STREAM_FLOW_ACCUM_THRESHOLD = float(
    os.environ.get('STREAM_FLOW_ACCUM_THRESHOLD', 1000)
)
TRIAL_WORKERS = int(os.environ.get('TRIAL_WORKERS', 4))
DEM_TILES = Path(os.environ.get('DEM_TILES', DATA_DIR / 'dem_tiles.shp'))
N_COLS = int(os.environ.get('N_COLS', -1))
LOG_INTERVAL = float(os.environ.get('LOG_INTERVAL', 0.25))
MIN_GRADIENT = float(os.environ.get('N_COLS', 0.01))
