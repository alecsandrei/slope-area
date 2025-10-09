from __future__ import annotations

from pathlib import Path

from slope_area.hydro import HydrologicAnalysis

TEST_DATA_DIR = Path(__file__).parent / 'data'


def test_hydrologic_analysis_process():
    dem = TEST_DATA_DIR / 'dem.tif'
    outlet = TEST_DATA_DIR / 'outlet.shp'
    HydrologicAnalysis(dem, outlet).process()
