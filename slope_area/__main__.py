from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from slope_area._internal.dem import (
    VRT,
    DEMSource,
    DEMTiles,
    DEMTilesBuilder,
    DynamicVRT,
    GeneralizedDEM,
)
from slope_area.builder import (
    BuilderConfig,
    OutletPlotBuilder,
    ResolutionPlotBuilder,
)
from slope_area.features import Outlets
from slope_area.geomorphometry import HydrologicAnalysisConfig
from slope_area.logger import create_logger
from slope_area.paths import PROJ_ROOT
from slope_area.plot import SlopeAreaPlotConfig

logger = create_logger(__name__)


def main() -> None:
    logger.info(
        'Running __main__ script. This will only work with access to internal data.'
    )
    # ---- Paths ----
    outlets_path = PROJ_ROOT / 'data' / 'raw' / 'outlets.shp'

    out_dir = PROJ_ROOT / 'data' / 'processed' / '__main__'
    out_dir_outlet_builder = out_dir / 'outlet_builder'
    out_dir_resolution_builder = out_dir / 'resolution_builder'

    out_fig_outlet_builder = out_dir_outlet_builder / 'slope_area.png'
    out_fig_resolution_builder = out_dir_resolution_builder / 'slope_area.png'

    # Internal data
    dem_dir = PROJ_ROOT / 'data' / 'raw' / 'DEM'
    dem_tiles_path = PROJ_ROOT / 'data' / 'raw' / 'dem_tiles.fgb'
    generalized_dem_path = PROJ_ROOT / 'data' / 'raw' / 'dem_30m.tif'
    generalized_dem_out = out_dir / generalized_dem_path.stem

    # ---- Run configs ----
    dem_dir_epsg = 3844
    max_workers = 3
    hydrologic_analysis_config = HydrologicAnalysisConfig(
        streams_flow_accumulation_threshold=1000, outlet_snap_distance=100
    )
    plot_config = SlopeAreaPlotConfig(
        hue='slope_type',
        col='trial',
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
    resolutions = [(res, res) for res in range(5, 15)]
    outlet_name = 'gully 2'

    builder_config = BuilderConfig(
        hydrologic_analysis_config=hydrologic_analysis_config,
        out_dir=out_dir_resolution_builder,
        out_fig=out_fig_resolution_builder,
        plot_config=plot_config,
        max_workers=max_workers,
    )

    outlet = [outlet for outlet in outlets if outlet.name == outlet_name][0]

    dem = Path(builder_config.out_dir) / 'dem.vrt'
    dem_tiles = DEMTiles.from_outlet(
        dem_source=dem_source,
        outlet=outlet,
        out_dir=builder_config.out_dir,
        outlet_snap_dist=builder_config.hydrologic_analysis_config.outlet_snap_distance,
    )
    vrt = VRT.from_dem_tiles(dem_tiles, dem).define_projection(dem_source.crs)

    _ = ResolutionPlotBuilder(builder_config, vrt, outlet, resolutions).build()

    # ---- Plot comparing different outlets ----
    resolution = (5, 5)

    builder_config = BuilderConfig(
        hydrologic_analysis_config=hydrologic_analysis_config,
        out_dir=out_dir_outlet_builder,
        out_fig=out_fig_outlet_builder,
        plot_config=plot_config,
        max_workers=max_workers,
    )

    # Computing the DEM for each outlet is a heavy process with the internal dataset.
    # The DynamicVRT allows to compute the VRT from DEMTiles in the process of each Trial
    # DynamicVRT follows the DEMProvider interface
    dem_provider = DynamicVRT(
        dem_source, out_dir_outlet_builder, outlet_snap_distance=100
    )

    _ = OutletPlotBuilder(
        builder_config, dem=dem_provider, outlets=outlets, resolution=resolution
    ).build()


if __name__ == '__main__':
    main()
