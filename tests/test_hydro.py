from __future__ import annotations

from pathlib import Path

from slope_area.geomorphometry import HydrologicAnalysis

TEST_DATA_DIR = Path(__file__).parent / 'data'


def test_hydrologic_analysis_process(tmpdir: Path):
    dem = TEST_DATA_DIR / 'dem.tif'
    outlet = TEST_DATA_DIR / 'outlet.shp'
    HydrologicAnalysis(dem, tmpdir).compute_slope_gradient(
        outlet, streams_flow_accum_threshold=100, outlet_snap_dist=100
    )
