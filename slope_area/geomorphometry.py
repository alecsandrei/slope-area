from __future__ import annotations

import collections.abc as c
from dataclasses import dataclass, field
from functools import cache, partial
import logging
from os import PathLike, fspath, makedirs
from pathlib import Path
import reprlib
import typing as t

from PySAGA_cmd import Raster as SAGARaster
from PySAGA_cmd import Vector as SAGAVector
from whitebox_workflows.whitebox_workflows import Raster as WhiteboxRaster
from whitebox_workflows.whitebox_workflows import Vector as WhiteboxVector

from slope_area import SAGA_ENV, WBW_ENV
from slope_area.config import (
    INTERIM_DATA_DIR,
    STREAM_FLOW_ACCUM_THRESHOLD,
)
from slope_area.logger import create_logger
from slope_area.utils import (
    suppress_stdout_stderr,
    timeit,
    write_whitebox,
)

if t.TYPE_CHECKING:
    from whitebox_workflows.whitebox_workflows import WbEnvironment

logger = create_logger(__name__)


@dataclass(frozen=True)
class ComputationOutput:
    def write_whitebox(
        self,
        out_dir: Path,
        prefix: str = '',
        names: c.Sequence[str] | None = None,
        *,
        logger: logging.Logger | None = None,
        recurse: bool = False,
        overwrite: bool = False,
        wbw_env: WbEnvironment = WBW_ENV,
    ):
        if logger is None:
            logger = create_logger(__file__)
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


@dataclass
class HydrologicAnalysis:
    dem: WhiteboxRaster
    out_dir: Path
    wbw_env: WbEnvironment
    _logger: logging.Logger = field(init=False, repr=False)

    def __init__(
        self,
        dem: PathLike | WhiteboxRaster,
        out_dir: PathLike = INTERIM_DATA_DIR,
        wbw_env: WbEnvironment = WBW_ENV,
    ):
        self._logger = logger.getChild(self.__class__.__name__)
        self.wbw_env = wbw_env
        if not isinstance(dem, WhiteboxRaster):
            self._logger.info('Reading the DEM at %s.' % dem)
            dem = wbw_env.read_raster(fspath(dem))
        self.dem = dem
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True)

    @timeit(logger, logging.DEBUG)
    def preprocess_dem(self) -> WhiteboxRaster:
        self._logger.info('Breaching depressions in the DEM.')
        dem_preproc = self.wbw_env.breach_depressions_least_cost(
            dem=self.dem, fill_deps=True
        )
        self._logger.info('Breaching single-cell pits.')
        dem_preproc = self.wbw_env.breach_single_cell_pits(dem_preproc)
        return dem_preproc

    @timeit(logger, logging.DEBUG)
    def compute_flow(
        self, dem_preproc: WhiteboxRaster | None = None
    ) -> FlowAccumulationComputationOutput:
        dem_preproc = (
            dem_preproc if dem_preproc is not None else self.preprocess_dem()
        )
        self._logger.info('Computing the D8 pointer.')
        d8_pointer = self.wbw_env.d8_pointer(dem_preproc)
        self._logger.info('Computing the flow accumulation.')
        with suppress_stdout_stderr():
            flow = self.wbw_env.d8_flow_accum(
                d8_pointer, out_type='catchment_area', input_is_pointer=True
            )
        return FlowAccumulationComputationOutput(dem_preproc, d8_pointer, flow)

    @timeit(logger, logging.DEBUG)
    def compute_watershed(
        self,
        outlet: WhiteboxVector | PathLike,
        outlet_snap_dist: int | None = None,
        dem_preproc: WhiteboxRaster | None = None,
    ) -> WatershedComputationOutput:
        if isinstance(outlet, PathLike):
            self._logger.info('Reading the outlet %s.' % outlet)
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
        self._logger.info('Computing the watershed.')
        watershed = self.wbw_env.watershed(flow_output.d8_pointer, outlet)
        return WatershedComputationOutput(flow_output, outlet, watershed)

    @timeit(logger, logging.DEBUG)
    def compute_slope_gradient(
        self,
        outlet: WhiteboxVector | PathLike,
        streams_flow_accum_threshold: float = STREAM_FLOW_ACCUM_THRESHOLD,
        outlet_snap_dist: float | None = None,
    ) -> SlopeGradientComputationOutput:
        write_whitebox_func = partial(
            write_whitebox,
            logger=self._logger,
            overwrite=True,
            wbw_env=self.wbw_env,
        )
        if isinstance(outlet, PathLike):
            self._logger.info('Reading the outlet %s.' % outlet)
            outlet = self.wbw_env.read_vector(fspath(outlet))

        # ---- Computing D8 pointer and flow accumulation ----
        watershed_output = self.compute_watershed(outlet, outlet_snap_dist)
        write_whitebox_func(
            watershed_output.flow.dem_preproc,
            self.out_dir / 'dem_preproc.tif',
        )
        write_whitebox_func(
            watershed_output.watershed,
            self.out_dir / 'watershed.tif',
        )

        # ---- Using the watershed to mask the DEM ----
        self._logger.info('Masking the preprocessed DEM with the watershed.')
        dem_preproc_mask = watershed_output.watershed.con(
            'value == 1',
            true_raster_or_float=watershed_output.flow.dem_preproc,
            false_raster_or_float=watershed_output.watershed,
        )

        # ---- Computing the slope gradient for the masked DEM ----
        self._logger.info(
            'Computing the slope gradient for the streams in the watershed.'
        )
        flow_watershed = self.compute_flow(dem_preproc_mask)
        streams = self.wbw_env.find_main_stem(
            flow_watershed.d8_pointer,
            self.wbw_env.extract_streams(
                flow_watershed.flow_accumulation, streams_flow_accum_threshold
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
            self.out_dir / 'dem_preproc_mask.tif',
        )
        write_whitebox_func(
            slope_gradient_output.flow.flow_accumulation,
            self.out_dir / 'flowacc.tif',
        )
        write_whitebox_func(
            slope_gradient_output.slope_grad,
            self.out_dir / 'slope_grad.tif',
        )
        return slope_gradient_output


def compute_slope(elevation: PathLike, out_slope: PathLike) -> SAGARaster:
    tool = SAGA_ENV / 'ta_morphometry' / 'Slope, Aspect, Curvature'
    logger.info('Computing slope for DEM %s.' % elevation)
    return tool.execute(
        verbose=False,
        elevation=elevation,
        slope=out_slope,
        unit_slope=2,  # Percent rise
        method=6,  # 9 parameter 2nd order polynom (Zevenbergen & Thorne 1987)
    ).rasters['slope']


def grid_values_to_points(
    rasters: c.Sequence[PathLike | str],
    points: PathLike | str,
) -> SAGAVector:
    profiles_from_lines = SAGA_ENV / 'shapes_grid' / 'Add Grid Values to Points'
    rasters_str = ';'.join(fspath(raster) for raster in rasters)
    logger.info(
        'Extracting points for rasters %s.'
        % ', '.join([reprlib.repr(fspath(raster)) for raster in rasters])
    )
    return profiles_from_lines.execute(
        verbose=False,
        grids=rasters_str,
        shapes=points,
    )


def compute_profile_from_lines(
    rasters: c.Sequence[PathLike | str],
    lines: PathLike | str,
    out_profile: PathLike,
    dem: PathLike | str | None = None,
    split_by_field: str = 'FID',
) -> SAGAVector:
    if dem is None:
        dem = rasters[0]
    profiles_from_lines = SAGA_ENV / 'ta_profiles' / 'Profiles from Lines'
    rasters_str = ';'.join(fspath(raster) for raster in rasters)
    logger.info(
        'Computing profile for rasters %s.'
        % ', '.join([reprlib.repr(fspath(raster)) for raster in rasters])
    )
    return profiles_from_lines.execute(
        verbose=False,
        dem=dem,
        values=rasters_str,
        lines=lines,
        profile=out_profile,
        name=split_by_field,
    ).vectors['profile']


def degree_to_percent(raster: WhiteboxRaster) -> WhiteboxRaster:
    return raster.to_radians().tan() * 100


@dataclass(frozen=True, eq=True)
class GeneralizedDEM:
    dem: PathLike
    out_dir: PathLike

    @property
    def dem_preproc(self) -> WhiteboxRaster:
        return self.get_flow_output().dem_preproc

    @property
    def d8_pointer(self) -> WhiteboxRaster:
        return self.get_flow_output().d8_pointer

    @property
    def flow_accum(self) -> WhiteboxRaster:
        return self.get_flow_output().flow_accumulation

    @timeit(logger, level=logging.DEBUG)
    def read_rasters(self, rasters: c.Iterable[PathLike]):
        # also tried read_rasters but it panics
        return [WBW_ENV.read_raster(fspath(raster)) for raster in rasters]

    @cache
    def get_flow_output(self) -> FlowAccumulationComputationOutput:
        out_dir = Path(self.out_dir)
        rasters = (
            out_dir / 'dem_preproc.tif',
            out_dir / 'd8_pointer.tif',
            out_dir / 'flow_accumulation.tif',
        )
        if all(raster.exists() for raster in rasters):
            wbw_rasters = self.read_rasters(rasters)
            return FlowAccumulationComputationOutput(*wbw_rasters)
        else:
            hydro_analysis = HydrologicAnalysis(self.dem, out_dir=self.out_dir)
            output = hydro_analysis.compute_flow()
            output.write_whitebox(
                out_dir,
                overwrite=True,
                logger=logger.getChild(self.__class__.__name__),
            )
            return output
