from __future__ import annotations

import geopandas as gpd

from slope_area.builder import Trial, TrialConfig
from slope_area.config import (
    PROJ_ROOT,
)
from slope_area.features import Outlets
from slope_area.geomorphometry import HydrologicAnalysisConfig
from slope_area.logger import create_logger

logger = create_logger(__name__)


def main():
    outlet_name = 'gully 2'
    outlets_path = PROJ_ROOT / 'data' / 'raw' / 'outlets.shp'
    logger.info('Reading outlets at %s' % outlets_path)
    gdf = gpd.read_file(outlets_path)
    outlets = Outlets.from_gdf(gdf, name_field='name')
    outlet = [outlet for outlet in outlets if outlet.name == outlet_name][0]
    trial_config = TrialConfig(
        name=outlet_name,
        outlet=outlet,
        resolution=(5, 5),
        hydrologic_analysis_config=HydrologicAnalysisConfig(
            streams_flow_accumulation_threshold=100, outlet_snap_distance=100
        ),
        out_dir=PROJ_ROOT / 'data' / 'processed' / outlet_name,
    )
    Trial(trial_config).run()


if __name__ == '__main__':
    main()
