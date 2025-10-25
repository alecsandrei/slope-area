from __future__ import annotations

import collections.abc as c
from dataclasses import InitVar, dataclass, field
from functools import partial
import logging
from os import PathLike, fspath, makedirs
from pathlib import Path
import typing as t

from PySAGA_cmd import Raster as SAGARaster
from whitebox_workflows.whitebox_workflows import Raster as WhiteboxRaster
from whitebox_workflows.whitebox_workflows import Vector as WhiteboxVector

from slope_area import SAGA_ENV, WBW_ENV
from slope_area._typing import AnyLogger
from slope_area.logger import create_logger
from slope_area.utils import (
    suppress_stdout_stderr,
    timeit,
    write_whitebox,
)

if t.TYPE_CHECKING:
    from whitebox_workflows.whitebox_workflows import WbEnvironment

m_logger = create_logger(__name__)


@dataclass(frozen=True)
class ComputationOutput:
    def write_whitebox(
        self,
        out_dir: Path,
        prefix: str = '',
        names: c.Sequence[str] | None = None,
        *,
        logger: AnyLogger = m_logger,
        recurse: bool = False,
        overwrite: bool = False,
        wbw_env: WbEnvironment = WBW_ENV,
    ):
        makedirs(out_dir, exist_ok=True)
        for name, output in self.__dict__.items():
            if names is not None and name not in names:
                continue
            output = self.__dict__[name]
            out_base = out_dir / f'{prefix}{name}'
            out_file = None
            if isinstance(output, WhiteboxRaster):
                out_file = out_base.with_suffix('.tif')
            elif isinstance(output, WhiteboxVector):
                out_file = out_base.with_suffix('.shp')
            elif isinstance(output, ComputationOutput):
                if recurse:
                    output.write_whitebox(
                        out_dir=out_dir / name,
                        prefix=prefix,
                        recurse=recurse,
                        wbw_env=wbw_env,
                    )
                continue
            else:
                logger.error(
                    'Failed to save %s. Unknown Whitebox Workflows object %s'
                    % (name, output)
                )
                continue
            write_whitebox(
                output,
                out_file,
                logger=logger,
                overwrite=overwrite,
                wbw_env=wbw_env,
            )


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
class SlopeGradientComputationOutput(ComputationOutput):
    watershed: WatershedComputationOutput
    flow: FlowAccumulationComputationOutput
    streams: WhiteboxRaster
    slope_grad: WhiteboxRaster


@dataclass(frozen=True)
class HydrologicAnalysisConfig:
    streams_flow_accumulation_threshold: int = 1000
    outlet_snap_distance: int = 100


@dataclass
class HydrologicAnalysis:
    dem: PathLike | WhiteboxRaster
    out_dir: PathLike
    wbw_env: WbEnvironment = field(repr=False, kw_only=True, default=WBW_ENV)
    logger: InitVar[AnyLogger | None] = field(kw_only=True, default=None)
    _logger: AnyLogger = field(init=False, repr=False)

    def __post_init__(self, logger: AnyLogger | None):
        self._logger = logger or m_logger.getChild(self.__class__.__name__)
        makedirs(self.out_dir, exist_ok=True)

    @timeit(m_logger, logging.DEBUG)
    def read_dem(self) -> WhiteboxRaster:
        if not isinstance(self.dem, WhiteboxRaster):
            self._logger.info('Reading DEM %s' % Path(self.dem).name)
            return self.wbw_env.read_raster(fspath(self.dem))
        return self.dem

    @timeit(m_logger, logging.DEBUG)
    def preprocess_dem(self) -> WhiteboxRaster:
        self._logger.info('Breaching depressions in the DEM')
        dem_preproc = self.wbw_env.breach_depressions_least_cost(
            dem=self.read_dem(), fill_deps=True
        )
        self._logger.info('Breaching single-cell pits')
        dem_preproc = self.wbw_env.breach_single_cell_pits(dem_preproc)
        return dem_preproc

    @timeit(m_logger, logging.DEBUG)
    def compute_flow(
        self, dem_preproc: WhiteboxRaster | None = None
    ) -> FlowAccumulationComputationOutput:
        dem_preproc = (
            dem_preproc if dem_preproc is not None else self.preprocess_dem()
        )
        self._logger.info('Computing the D8 pointer')
        d8_pointer = self.wbw_env.d8_pointer(dem_preproc)
        self._logger.info('Computing the flow accumulation')
        with suppress_stdout_stderr():
            flow = self.wbw_env.d8_flow_accum(
                d8_pointer, out_type='catchment_area', input_is_pointer=True
            )
        return FlowAccumulationComputationOutput(dem_preproc, d8_pointer, flow)

    @timeit(m_logger, logging.DEBUG)
    def compute_watershed(
        self,
        outlet: WhiteboxVector | PathLike,
        outlet_snap_dist: int | None = None,
        dem_preproc: WhiteboxRaster | None = None,
    ) -> WatershedComputationOutput:
        if isinstance(outlet, PathLike):
            self._logger.info('Reading the outlet %s' % outlet)
            outlet = self.wbw_env.read_vector(fspath(outlet))
        flow_output = self.compute_flow(dem_preproc)
        if outlet_snap_dist:
            self._logger.info(
                'Snapping the outlet using a snap distance of %.1f'
                % outlet_snap_dist
            )
            outlet = self.wbw_env.snap_pour_points(
                outlet, flow_output.flow_accumulation, outlet_snap_dist
            )
        self._logger.info('Computing the watershed')
        watershed = self.wbw_env.watershed(flow_output.d8_pointer, outlet)
        return WatershedComputationOutput(flow_output, outlet, watershed)

    @timeit(m_logger, logging.DEBUG)
    def compute_slope_gradient(
        self,
        outlet: WhiteboxVector | PathLike,
        config: HydrologicAnalysisConfig,
    ) -> SlopeGradientComputationOutput:
        out_dir = Path(self.out_dir)
        write_whitebox_func = partial(
            write_whitebox,
            logger=self._logger,
            overwrite=True,
            wbw_env=self.wbw_env,
        )
        if isinstance(outlet, PathLike):
            self._logger.info('Reading the outlet %s' % outlet)
            outlet = self.wbw_env.read_vector(fspath(outlet))

        # ---- Computing D8 pointer and flow accumulation ----
        watershed_output = self.compute_watershed(
            outlet, config.outlet_snap_distance
        )

        write_whitebox_func(
            watershed_output.flow.dem_preproc,
            out_dir / 'dem_preproc.tif',
        )
        write_whitebox_func(
            watershed_output.watershed,
            out_dir / 'watershed.tif',
        )

        # ---- Using the watershed to mask the DEM ----
        self._logger.info('Masking the preprocessed DEM with the watershed')
        dem_preproc_mask = watershed_output.watershed.con(
            'value == 1',
            true_raster_or_float=watershed_output.flow.dem_preproc,
            false_raster_or_float=watershed_output.watershed,
        )

        # ---- Computing the slope gradient for the masked DEM ----
        self._logger.info(
            'Computing the slope gradient for the streams in the watershed'
        )
        flow_watershed = self.compute_flow(dem_preproc_mask)
        streams = self.wbw_env.find_main_stem(
            flow_watershed.d8_pointer,
            self.wbw_env.extract_streams(
                flow_watershed.flow_accumulation,
                config.streams_flow_accumulation_threshold,
            ),
        )
        slope_gradient = degree_to_percent(
            self.wbw_env.stream_slope_continuous(
                flow_watershed.d8_pointer, streams, dem_preproc_mask
            )
        )
        slope_gradient_output = SlopeGradientComputationOutput(
            watershed_output, flow_watershed, streams, slope_gradient
        )
        write_whitebox_func(
            slope_gradient_output.flow.dem_preproc,
            out_dir / 'dem_preproc_mask.tif',
        )
        write_whitebox_func(
            slope_gradient_output.flow.flow_accumulation,
            out_dir / 'flowacc.tif',
        )
        write_whitebox_func(
            slope_gradient_output.slope_grad,
            out_dir / 'slope_grad.tif',
        )
        return slope_gradient_output


def compute_slope(
    elevation: PathLike, out_slope: PathLike, *, logger: AnyLogger = m_logger
) -> SAGARaster:
    tool = SAGA_ENV / 'ta_morphometry' / 'Slope, Aspect, Curvature'
    logger.info('Computing slope for DEM %s' % Path(elevation).name)
    return tool.execute(
        verbose=False,
        elevation=elevation,
        slope=out_slope,
        unit_slope=2,  # Percent rise
        method=6,  # 9 parameter 2nd order polynom (Zevenbergen & Thorne 1987)
    ).rasters['slope']


def degree_to_percent(raster: WhiteboxRaster) -> WhiteboxRaster:
    return raster.to_radians().tan() * 100
