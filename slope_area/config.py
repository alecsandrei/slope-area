from __future__ import annotations

import os
from pathlib import Path

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
LOGGING_CONFIG = PROJ_ROOT / 'logging' / 'config.json'
DATA_DIR = PROJ_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
INTERIM_DATA_DIR = DATA_DIR / 'interim'

# Raw data files
DEM_DIR = RAW_DATA_DIR / 'DEM'
DEM_TILES = RAW_DATA_DIR / 'dem_tiles.fgb'
DEM_90M = RAW_DATA_DIR / 'dem_90m.tif'
DEM_30M = RAW_DATA_DIR / 'dem_30m.tif'

# Environment varialbes
WORKERS = int(os.getenv('WORKERS', os.cpu_count() or 1))
