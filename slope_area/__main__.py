from __future__ import annotations

import geopandas as gpd

from slope_area.builder import ResolutionPlotBuilder
from slope_area.config import (
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
)
from slope_area.features import Outlets


def main():
    outlet_name = 'gully 2'
    resolutions = [(res, res) for res in range(1, 11)]
    outlets = Outlets.from_gdf(
        gpd.read_file(RAW_DATA_DIR / 'outlets.shp'), name_field='name'
    )
    outlet = [outlet for outlet in outlets if outlet.name == outlet_name]
    ResolutionPlotBuilder(
        outlet[0],
        resolutions,
        out_dir=INTERIM_DATA_DIR / outlet_name,
    ).build()


if __name__ == '__main__':
    main()
