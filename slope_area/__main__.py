from __future__ import annotations

import geopandas as gpd

from slope_area.builder import (
    BuilderConfig,
    DEMSource,
    OutletPlotBuilder,
    ResolutionPlotBuilder,
    StaticVRT,
    Trial,
    TrialConfig,
)
from slope_area.features import (
    VRT,
    DEMTiles,
    DEMTilesBuilder,
    GeneralizedDEM,
    Outlets,
)
from slope_area.geomorphometry import HydrologicAnalysisConfig
from slope_area.logger import create_logger
from slope_area.paths import PROJ_ROOT

logger = create_logger(__name__)


def main():
    # ---- Paths ----
    dem_dir = PROJ_ROOT / 'data' / 'raw' / 'DEM'
    dem_tiles = PROJ_ROOT / 'data' / 'raw' / 'dem_tiles.fgb'
    generalized_dem = PROJ_ROOT / 'data' / 'raw' / 'dem_30m.tif'
    generalized_dem_out = (
        PROJ_ROOT / 'data' / 'processed' / generalized_dem.stem
    )
    outlets = PROJ_ROOT / 'data' / 'raw' / 'outlets.shp'
    dem_dir_epsg = 3844
    out_dir = PROJ_ROOT / 'data' / 'processed'

    # ---- Run configs ----
    plot_kind = 'resolution'
    outlet_name = 'gully 13'
    resolutions = [(res, res) for res in range(5, 15)]

    # ---- Init objects ----
    tiles = DEMTilesBuilder(
        dem_dir, dem_dir_epsg=dem_dir_epsg, tiles=dem_tiles
    ).build()
    generalized_dem = GeneralizedDEM(
        path=generalized_dem, out_dir=generalized_dem_out
    )
    dem_source = DEMSource(dem_dir, tiles, generalized_dem)
    hydrologic_analysis_config = HydrologicAnalysisConfig(
        streams_flow_accumulation_threshold=1000, outlet_snap_distance=100
    )

    logger.info('Reading outlets at %s' % outlets)
    gdf = gpd.read_file(outlets).sort_values(by='name')
    gdf = gdf[gdf['is_gully'] == 1]
    outlets = Outlets.from_gdf(gdf, name_field='name')

    if plot_kind == 'resolution':
        builder_config = BuilderConfig(
            hydrologic_analysis_config, out_dir / outlet_name, max_workers=4
        )
        outlet = [outlet for outlet in outlets if outlet.name == outlet_name]
        ResolutionPlotBuilder(
            builder_config, dem_source, outlet[0], resolutions
        ).build()
    elif plot_kind == 'outlet':
        builder_config = BuilderConfig(
            hydrologic_analysis_config, out_dir, max_workers=2
        )
        OutletPlotBuilder(
            builder_config, dem_source, outlets=outlets, resolution=(5, 5)
        ).build()
    else:
        outlet = outlets[0]
        out_dir = out_dir / outlet.name
        trial = Trial(
            TrialConfig(
                name=outlet.name,
                outlet=outlets[0],
                resolution=(10, 10),
                hydrologic_analysis_config=hydrologic_analysis_config,
                dem_provider=StaticVRT(
                    VRT.from_dem_tiles(
                        DEMTiles.from_outlet(
                            dem_source,
                            outlet,
                            out_dir,
                            hydrologic_analysis_config.outlet_snap_distance,
                        ),
                        out_dir / 'dem.vrt',
                    )
                ),
                out_dir=out_dir,
            )
        ).run()


if __name__ == '__main__':
    main()
