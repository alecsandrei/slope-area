from __future__ import annotations

from pathlib import Path

from slope_area.geomorphometry import (
    HydrologicAnalysis,
    HydrologicAnalysisConfig,
)

TEST_DATA_DIR = Path(__file__).parent / 'data'


def test_hydrologic_analysis_process(tmpdir: Path):
    dem = TEST_DATA_DIR / 'dem.tif'
    outlet = TEST_DATA_DIR / 'outlet.shp'
    HydrologicAnalysis(dem, tmpdir).compute_slope_gradient(
        outlet,
        HydrologicAnalysisConfig(
            streams_flow_accumulation_threshold=100, outlet_snap_distance=100
        ),
    )
