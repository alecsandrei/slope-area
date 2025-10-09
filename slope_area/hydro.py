from __future__ import annotations

from dataclasses import dataclass
import logging
from os import fspath
import typing as t

import geopandas as gpd

from slope_area import SAGA_ENV, WBE_ENV
from slope_area.config import INTERIM_DATA_DIR, RAW_DATA_DIR
from slope_area.logger import create_logger
from slope_area.utils import timeit

if t.TYPE_CHECKING:
    from os import PathLike

    from whitebox_workflows.whitebox_workflows import Raster as WhiteboxRaster
    from whitebox_workflows.whitebox_workflows import Vector as WhiteboxVector

logger = create_logger(__file__)


@dataclass(frozen=True)
class HydrologicAnalysisOutput:
    dem_preproc: WhiteboxRaster
    streams: WhiteboxVector
    flow_accumulation: WhiteboxRaster
    slope_gradient: WhiteboxRaster


@dataclass
class HydrologicAnalysis:
    dem: WhiteboxRaster
    outlet: WhiteboxVector
    streams_threshold: int
    outlet_snap_dist_units: int

    def __init__(
        self,
        dem: PathLike,
        outlet: PathLike,
        streams_threshold: int = 100,
        outlet_snap_dist_units: int = 15,
    ):
        logger.info('Reading the DEM and the outlet')
        self.dem = WBE_ENV.read_raster(fspath(dem))
        self.outlet = WBE_ENV.read_vector(fspath(outlet))
        self.streams_threshold = streams_threshold
        self.outlet_snap_dist_units = outlet_snap_dist_units

    @timeit(logger, logging.INFO)
    def process(self) -> HydrologicAnalysisOutput:
        logger.info('Breaching depressions in the DEM.')
        dem_preproc_path = INTERIM_DATA_DIR / 'dem_preproc.tif'
        dem_preproc = WBE_ENV.breach_depressions_least_cost(
            dem=self.dem, fill_deps=True
        )
        dem_preproc = WBE_ENV.breach_single_cell_pits(dem_preproc)
        WBE_ENV.write_raster(dem_preproc, dem_preproc_path.as_posix())

        logger.info('Computing the D8 pointer.')
        d8_path = INTERIM_DATA_DIR / 'd8.tif'
        d8 = WBE_ENV.d8_pointer(dem_preproc)
        WBE_ENV.write_raster(d8, d8_path.as_posix())

        logger.info('Computing the flow accumulation.')
        flow_path = INTERIM_DATA_DIR / 'flow.tif'
        flow = WBE_ENV.d8_flow_accum(d8, input_is_pointer=True)
        WBE_ENV.write_raster(flow, flow_path.as_posix())

        logger.info(
            'Extracting the streams using a flow accumulation threshold of %i.'
            % self.streams_threshold
        )
        streams_path = INTERIM_DATA_DIR / 'streams.tif'
        streams = WBE_ENV.extract_streams(
            flow, threshold=self.streams_threshold
        )
        WBE_ENV.write_raster(streams, streams_path.as_posix())

        logger.info(
            'Snapping the outlet to the pixel with the highest flow accumulation within %i units.'
            % self.outlet_snap_dist_units
        )
        outlet_snapped_path = INTERIM_DATA_DIR / 'pour_point_snapped.shp'
        outlet_snapped = WBE_ENV.snap_pour_points(
            self.outlet, flow, snap_dist=self.outlet_snap_dist_units
        )
        WBE_ENV.write_vector(outlet_snapped, outlet_snapped_path.as_posix())

        logger.info('Generating the watershed.')
        watershed_path = INTERIM_DATA_DIR / 'watershed.tif'
        watershed = WBE_ENV.watershed(d8, outlet_snapped)
        WBE_ENV.write_raster(watershed, watershed_path.as_posix())

        # logger.info('Converting watershed to vector')
        # watershed_as_vec_path = INTERIM_DATA_DIR / 'watershed.shp'
        # watershed_as_vec = WBE_ENV.raster_to_vector_polygons(watershed)
        # WBE_ENV.write_vector(watershed_as_vec, watershed_as_vec_path.as_posix())

        logger.info('Masking the preprocessed DEM with the watershed.')
        dem_preproc_mask_path = INTERIM_DATA_DIR / 'dem_preproc_mask.tif'
        raster_masking_tool = SAGA_ENV / 'grid_tools' / 'Grid Masking'
        output = (
            raster_masking_tool.execute(
                grid=dem_preproc_path,
                mask=watershed_path,
                masked=dem_preproc_mask_path,
            )
            .rasters['masked']
            .path
        )
        dem_preproc_mask = WBE_ENV.read_raster(fspath(output))

        logger.info('Computing the D8 pointer for the watershed.')
        d8_watershed_path = INTERIM_DATA_DIR / 'd8_watershed.tif'
        d8_watershed = WBE_ENV.d8_pointer(dem_preproc_mask)
        WBE_ENV.write_raster(d8_watershed, d8_watershed_path.as_posix())

        logger.info('Computing the flow accumulation for the watershed.')
        flow_watershed_path = INTERIM_DATA_DIR / 'flow_watershed.tif'
        flow_watershed = WBE_ENV.d8_flow_accum(
            d8_watershed, input_is_pointer=True, out_type='catchment area'
        )
        WBE_ENV.write_raster(flow_watershed, flow_watershed_path.as_posix())

        logger.info(
            'Extracting the streams for the watershed using a flow accumulation threshold of %i.'
            % self.streams_threshold
        )
        streams_path = INTERIM_DATA_DIR / 'streams.tif'
        streams = WBE_ENV.extract_streams(
            flow_watershed, threshold=self.streams_threshold
        )
        WBE_ENV.write_raster(streams, streams_path.as_posix())

        logger.info('Converting the raster stream network to vector.')
        streams_vector_path = INTERIM_DATA_DIR / 'streams.shp'
        streams_vector = WBE_ENV.raster_streams_to_vector(streams, d8_watershed)
        WBE_ENV.write_vector(streams_vector, streams_vector_path.as_posix())

        logger.info('Extracting the main stream from the stream network.')
        main_stream_path = INTERIM_DATA_DIR / 'main_stream_all_area.tif'
        main_stream = WBE_ENV.find_main_stem(d8_watershed, streams)
        WBE_ENV.write_raster(main_stream, main_stream_path.as_posix())

        logger.info('Converting the main stream to vector.')
        stream_path = INTERIM_DATA_DIR / 'main_stream.shp'
        main_stream_as_vec = WBE_ENV.raster_to_vector_lines(main_stream)
        WBE_ENV.write_vector(main_stream_as_vec, stream_path.as_posix())
        gpd.read_file(stream_path).iloc[::-1].dissolve().normalize().to_file(
            stream_path
        )

        logger.info(
            'Computing the slope gradient for the stream network/main stream.'
        )
        slope_continuous_path = INTERIM_DATA_DIR / 'slope_continuous_path.tif'
        slope_continuous = WBE_ENV.stream_slope_continuous(
            d8_watershed, streams, dem_preproc_mask
        )
        WBE_ENV.write_raster(slope_continuous, slope_continuous_path.as_posix())

        return HydrologicAnalysisOutput(
            dem_preproc_mask,
            main_stream_as_vec,
            flow_watershed,
            slope_continuous,
        )


if __name__ == '__main__':
    gully_number = '2'
    dem = RAW_DATA_DIR / 'ravene' / gully_number / '5 m' / 'merged.tif'
    outlet = RAW_DATA_DIR / 'ravene' / gully_number / 'pour_point.shp'
    print(HydrologicAnalysis(dem, outlet).process())
