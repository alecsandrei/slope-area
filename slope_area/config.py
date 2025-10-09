from __future__ import annotations

from pathlib import Path

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
LOGGING_CONFIG = PROJ_ROOT / 'logging' / 'config.json'
DATA_DIR = PROJ_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
INTERIM_DATA_DIR = DATA_DIR / 'interim'
