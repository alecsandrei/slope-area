from __future__ import annotations

from dataclasses import dataclass
import logging
from os import fspath
from pathlib import Path
import typing as t

from whitebox_workflows.whitebox_workflows import Raster as WhiteboxRaster
from whitebox_workflows.whitebox_workflows import Vector as WhiteboxVector

from slope_area._typing import SlopeProvider, StreamSlopeProvider
from slope_area.config import get_saga_env, get_wbw_env
from slope_area.logger import create_logger
from slope_area.utils import (
    read_whitebox_raster,
    read_whitebox_vector,
    suppress_stdout_stderr,
    timeit,
    write_whitebox,
)

if t.TYPE_CHECKING:
    from slope_area._typing import AnyLogger, StrPath

m_logger = create_logger(__name__)


class DefaultSlopeProviders:
    @dataclass
    class SAGASlope(SlopeProvider):
        method: int = 6

        def get_slope(self, dem: StrPath, out_file: StrPath) -> Path:
            return compute_3x3_slope(dem, out_file, method=self.method)

    @dataclass
    class StreamSlopeContinuous(StreamSlopeProvider):
        def get_stream_slope(
            self,
            streams: StreamsComputationOutput,
            out_file: StrPath,
        ) -> Path:
            output = compute_slope_gradient(streams)
            write_whitebox(output.slope_gradient, out_file, overwrite=True)
            return Path(out_file)


@dataclass(frozen=True)
class ComputationOutput: ...


@dataclass(frozen=True)
class FlowAccumulationComputationOutput(ComputationOutput):
    dem_preproc: WhiteboxRaster
    d8_pointer: WhiteboxRaster
    flow_accumulation: WhiteboxRaster


@dataclass(frozen=True)
class WatershedComputationOutput(ComputationOutput):
    flow: FlowAccumulationComputationOutput
    outlet: WhiteboxVector
    watershed: WhiteboxRaster


@dataclass(frozen=True)
class StreamsComputationOutput(ComputationOutput):
    flow: FlowAccumulationComputationOutput
    streams: WhiteboxRaster


@dataclass(frozen=True)
class SlopeGradientComputationOutput(ComputationOutput):
    streams: StreamsComputationOutput
    slope_gradient: WhiteboxRaster


@dataclass(frozen=True)
class HydrologicAnalysisConfig:
    streams_flow_accumulation_threshold: int = 1000
    outlet_snap_distance: int = 100


@timeit(m_logger, logging.INFO)
def preprocess_dem(
    dem: WhiteboxRaster | StrPath, *, logger: AnyLogger = m_logger
) -> WhiteboxRaster:
    wbw_env = get_wbw_env()
    dem = read_whitebox_raster(dem, logger=logger)
    logger.info('Breaching depressions in the DEM')
    dem_preproc = wbw_env.breach_depressions_least_cost(dem=dem, fill_deps=True)
    logger.info('Breaching single-cell pits')
    dem_preproc = wbw_env.breach_single_cell_pits(dem_preproc)
    return dem_preproc


@timeit(m_logger, logging.INFO)
def compute_flow(
    dem_preproc: WhiteboxRaster | StrPath, *, logger: AnyLogger = m_logger
) -> FlowAccumulationComputationOutput:
    wbw_env = get_wbw_env()
    dem_preproc = read_whitebox_raster(dem_preproc, logger=logger)
    logger.info('Computing the D8 pointer')
    d8_pointer = wbw_env.d8_pointer(dem_preproc)
    logger.info('Computing the flow accumulation')
    with suppress_stdout_stderr():
        flow = wbw_env.d8_flow_accum(
            d8_pointer, out_type='catchment_area', input_is_pointer=True
        )
    return FlowAccumulationComputationOutput(dem_preproc, d8_pointer, flow)


@timeit(m_logger, logging.INFO)
def compute_watershed(
    flow_accumulation_output: FlowAccumulationComputationOutput,
    outlet: WhiteboxVector | StrPath,
    outlet_snap_distance: float | None = None,
    *,
    logger: AnyLogger = m_logger,
) -> WatershedComputationOutput:
    logger.info('Computing watershed')
    wbw_env = get_wbw_env()
    outlet = read_whitebox_vector(outlet, logger=logger)
    if outlet_snap_distance:
        logger.info(
            'Snapping the outlet using a snap distance of %.1f'
            % outlet_snap_distance
        )
        outlet = wbw_env.snap_pour_points(
            outlet,
            flow_accumulation_output.flow_accumulation,
            outlet_snap_distance,
        )
    logger.info('Computing the watershed')
    watershed = wbw_env.watershed(flow_accumulation_output.d8_pointer, outlet)
    return WatershedComputationOutput(
        flow_accumulation_output, outlet, watershed
    )


@timeit(m_logger, logging.INFO)
def compute_streams(
    flow_accumulation_output: FlowAccumulationComputationOutput,
    flow_accumulation_threshold: float,
    main_stream: bool = True,
    *,
    logger: AnyLogger = m_logger,
) -> StreamsComputationOutput:
    logger.info(
        'Computing streams with flow accumulation threshold %.1f'
        % flow_accumulation_threshold
    )
    wbw_env = get_wbw_env()
    streams = wbw_env.extract_streams(
        flow_accumulation_output.flow_accumulation, flow_accumulation_threshold
    )
    if main_stream:
        streams = wbw_env.find_main_stem(
            flow_accumulation_output.d8_pointer,
            streams,
        )
    return StreamsComputationOutput(flow_accumulation_output, streams)


@timeit(m_logger, logging.INFO)
def compute_slope_gradient(
    streams_output: StreamsComputationOutput,
    *,
    logger: AnyLogger = m_logger,
) -> SlopeGradientComputationOutput:
    wbw_env = get_wbw_env()
    logger.info('Computing the slope gradient')
    slope_gradient = degree_to_percent(
        wbw_env.stream_slope_continuous(
            streams_output.flow.d8_pointer,
            streams_output.streams,
            streams_output.flow.dem_preproc,
        )
    )
    return SlopeGradientComputationOutput(streams_output, slope_gradient)


@timeit(m_logger, logging.INFO)
def compute_3x3_slope(
    elevation: StrPath,
    out_slope: StrPath,
    method: int = 6,  # 9 parameter 2nd order polynom (Zevenbergen & Thorne 1987)
    *,
    logger: AnyLogger = m_logger,
) -> Path:
    saga_env = get_saga_env()
    tool = saga_env / 'ta_morphometry' / 'Slope, Aspect, Curvature'
    logger.info(
        'Computing slope for elevation raster %s' % Path(elevation).name
    )
    return Path(
        tool.execute(
            verbose=False,
            elevation=fspath(elevation),
            slope=fspath(out_slope),
            unit_slope=2,  # Percent rise
            method=method,
        )
        .rasters['slope']
        .path
    )


def degree_to_percent(raster: WhiteboxRaster) -> WhiteboxRaster:
    return raster.to_radians().tan() * 100
