from __future__ import annotations

import geopandas as gpd

from slope_area._internal.dem import (
    VRT,
    DEMSource,
    DEMTiles,
    DEMTilesBuilder,
    DynamicVRT,
    GeneralizedDEM,
)
from slope_area.enums import Column, SlopeAreaMethod
from slope_area.features import Outlets
from slope_area.logger import create_logger
from slope_area.paths import PROJ_ROOT
from slope_area.plot import SlopeAreaPlotConfig
from slope_area.trial import (
    AnalysisConfig,
    HydrologicAnalysisConfig,
    OutletTrialFactory,
    ResolutionTrialFactory,
    TrialFactoryContext,
)

logger = create_logger(__name__)


def main() -> None:
    logger.info(
        'Running __main__ script. This will only work with access to internal data.'
    )
    # ---- Paths ----
    outlets_path = PROJ_ROOT / 'data' / 'raw' / 'outlets.shp'

    out_dir = PROJ_ROOT / 'data' / 'processed' / '__main__'
    out_dir_outlet = out_dir / 'outlet'
    out_dir_resolution = out_dir / 'resolution'

    out_fig_outlet = out_dir_outlet / 'slope_area.png'
    out_fig_resolution = out_dir_resolution / 'slope_area.png'

    # Internal data
    dem_dir = PROJ_ROOT / 'data' / 'raw' / 'DEM'
    dem_tiles_path = PROJ_ROOT / 'data' / 'raw' / 'dem_tiles.fgb'
    generalized_dem_path = PROJ_ROOT / 'data' / 'raw' / 'dem_30m.tif'
    generalized_dem_out = out_dir / generalized_dem_path.stem

    # ---- Run configs ----
    dem_dir_epsg = 3844
    max_workers = 5
    hydrologic_analysis_config = HydrologicAnalysisConfig(
        streams_flow_accumulation_threshold=1000, outlet_snap_distance=100
    )
    analysis_config = AnalysisConfig(
        method=SlopeAreaMethod.STREAMS, hydrologic=hydrologic_analysis_config
    )
    plot_config = SlopeAreaPlotConfig(
        hue=Column.SLOPE_TYPE,
        col=Column.TRIAL_NAME,
        log_interval=0.25,
        min_gradient=0.01,
        col_wrap=-1,
        height=5,
        aspect=1,
        title=None,
        xlabel='Drainage area (m$^2$)',
        ylabel='Slope (m/m)',
        label_font_size=16,
        title_font_size=10,
        legend_font_size=10,
        tick_font_size=14,
        add_vlines=False,
        kind='line',
        show=True,
    )

    # Internal objects for the Prut-Bârlad Water Administration LiDAR dataset
    tiles = DEMTilesBuilder(
        dem_dir, dem_dir_epsg=dem_dir_epsg, tiles=dem_tiles_path
    ).build()
    generalized_dem = GeneralizedDEM(
        path=generalized_dem_path, out_dir=generalized_dem_out
    )
    dem_source = DEMSource(dem_dir, tiles, generalized_dem, crs=3844)

    # ---- Read outlets ----
    logger.info('Reading outlets at %s' % outlets_path)
    gdf = gpd.read_file(outlets_path).sort_values(by='name')
    gdf = gdf[gdf['is_gully'] == 1]
    outlets = Outlets.from_gdf(gdf, name_field='name')

    # ---- Plot comparing different DEM resolutions ----
    resolutions = [(res, res) for res in range(1, 16)]
    outlet_name = 'gully 2'

    outlet = [outlet for outlet in outlets if outlet.name == outlet_name][0]

    dem = out_dir_resolution / 'dem.vrt'
    dem_tiles = DEMTiles.from_outlet(
        dem_source=dem_source,
        outlet=outlet,
        out_dir=dem.parent,
        outlet_snap_dist=hydrologic_analysis_config.outlet_snap_distance,
    )
    vrt = VRT.from_dem_tiles(dem_tiles, dem).define_projection(dem_source.crs)

    trials = ResolutionTrialFactory(
        context=TrialFactoryContext(
            dem=vrt, out_dir=out_dir_resolution, analysis=analysis_config
        ),
        outlet=outlet,
        resolutions=resolutions,
    ).generate()
    results = trials.run(max_workers=max_workers)
    results.plot(config=plot_config, out_fig=out_fig_resolution)

    # ---- Plot comparing different outlets ----
    resolution = (5, 5)

    # Computing the DEM for each outlet is a heavy process with the internal dataset.
    # The DynamicVRT allows to compute the VRT from DEMTiles in the process of each Trial
    # DynamicVRT follows the DEMProvider interface
    dem_provider = DynamicVRT(
        dem_source, out_dir_outlet, outlet_snap_distance=100
    )

    trials = OutletTrialFactory(
        context=TrialFactoryContext(
            dem=dem_provider, out_dir=out_dir_outlet, analysis=analysis_config
        ),
        outlets=outlets,
        resolution=resolution,
    ).generate()
    results = trials.run(max_workers=max_workers)
    results.plot(config=plot_config, out_fig=out_fig_outlet)


if __name__ == '__main__':
    main()
