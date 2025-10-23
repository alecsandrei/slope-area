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
GENERALIZED_DEM = Path(os.environ['GENERALIZED_DEM'])
SAGA_CMD = os.environ.get('SAGA_CMD', 'saga_cmd')
OUTLET_SNAP_DIST = float(os.environ.get('OUTLET_SNAP_DIST', 100))
TRIAL_WORKERS = int(os.environ.get('TRIAL_WORKERS', 4))

DEM_TILES = RAW_DATA_DIR / 'dem_tiles.fgb'
