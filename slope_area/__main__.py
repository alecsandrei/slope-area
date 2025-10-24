from __future__ import annotations

import geopandas as gpd

from slope_area.builder import OutletPlotBuilder, ResolutionPlotBuilder, Trial
from slope_area.config import (
    GENERALIZED_DEM,
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
)
from slope_area.features import DEMTilesBuilder, Outlets
from slope_area.geomorphometry import GeneralizedDEM
from slope_area.logger import create_logger

logger = create_logger(__name__)


def main():
    DEMTilesBuilder.build()
    plot = 'resolution'
    outlet_name = 'gully 12'
    generalized_dem = GeneralizedDEM(
        dem=GENERALIZED_DEM, out_dir=INTERIM_DATA_DIR / 'generalized_dem'
    )
    resolutions = [(res, res) for res in range(1, 30)]
    outlets_path = RAW_DATA_DIR / 'outlets.shp'
    logger.info('Reading outlets at %s.' % outlets_path)
    gdf = gpd.read_file(outlets_path).sort_values(by='name')
    outlets = Outlets.from_gdf(gdf, name_field='name')

    if plot == 'resolution':
        outlet = [outlet for outlet in outlets if outlet.name == outlet_name]
        ResolutionPlotBuilder(
            outlet[0],
            resolutions,
            generalized_dem=generalized_dem,
            out_dir=INTERIM_DATA_DIR / outlet_name,
        ).build()
    elif plot == 'outlet':
        OutletPlotBuilder(
            outlets,
            resolution=(20, 20),
            generalized_dem=generalized_dem,
            out_dir=INTERIM_DATA_DIR / 'outlets',
        ).build()
    else:
        Trial(
            [outlet for outlet in outlets if outlet.name == outlet_name][0],
            INTERIM_DATA_DIR / outlet_name,
            outlet_name,
            (10, 10),
        ).run()


if __name__ == '__main__':
    main()
