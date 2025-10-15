from __future__ import annotations

import collections.abc as c
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
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
    DATA_DIR,
    DEM_30M,
    DEM_90M,
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
)
from slope_area.logger import create_logger
from slope_area.utils import timeit, write_whitebox

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
                        out_dir=out_dir / name, prefix=prefix, recurse=recurse
                    )
                continue
            else:
                logger.error(
                    'Failed to save %s. Unknown Whitebox Workflows object %s'
                    % (name, output)
                )
                continue
            write_whitebox(output, out_file, logger=logger, overwrite=overwrite)


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
    _logger: logging.Logger = field(init=False, repr=False)

    def __init__(
        self,
        dem: PathLike | WhiteboxRaster,
        out_dir: PathLike = INTERIM_DATA_DIR,
    ):
        self._logger = logger.getChild(self.__class__.__name__)
        if not isinstance(dem, WhiteboxRaster):
            self._logger.info('Reading the DEM at %s.' % dem)
            dem = WBW_ENV.read_raster(fspath(dem))
        self.dem = dem
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True)

    @timeit(logger, logging.DEBUG)
    def preprocess_dem(self) -> WhiteboxRaster:
        self._logger.info('Breaching depressions in the DEM.')
        dem_preproc = WBW_ENV.breach_depressions_least_cost(
            dem=self.dem, fill_deps=True
        )
        self._logger.info('Breaching single-cell pits.')
        dem_preproc = WBW_ENV.breach_single_cell_pits(dem_preproc)
        return dem_preproc

    @timeit(logger, logging.DEBUG)
    def compute_flow(
        self, dem_preproc: WhiteboxRaster | None = None
    ) -> FlowAccumulationComputationOutput:
        dem_preproc = (
            dem_preproc if dem_preproc is not None else self.preprocess_dem()
        )
        self._logger.info('Computing the D8 pointer.')
        d8_pointer = WBW_ENV.d8_pointer(dem_preproc)
        self._logger.info('Computing the flow accumulation.')
        flow = WBW_ENV.d8_flow_accum(
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
            outlet = WBW_ENV.read_vector(fspath(outlet))
        flow_output = self.compute_flow(dem_preproc)
        if outlet_snap_dist:
            self._logger.info(
                'Snapping the outlet using a snap distance of %.1f'
                % outlet_snap_dist
            )
            outlet = WBW_ENV.snap_pour_points(
                outlet, flow_output.flow_accumulation, outlet_snap_dist
            )
        self._logger.info('Computing the watershed.')
        watershed = WBW_ENV.watershed(flow_output.d8_pointer, outlet)
        return WatershedComputationOutput(flow_output, outlet, watershed)

    @timeit(logger, logging.DEBUG)
    def compute_slope_gradient(
        self,
        outlet: WhiteboxVector | PathLike,
        streams_flow_accum_threshold: int = 100,
        outlet_snap_dist: int | None = None,
    ) -> SlopeGradientComputationOutput:
        write_whitebox_func = partial(
            write_whitebox, logger=self._logger, overwrite=True
        )
        if isinstance(outlet, PathLike):
            self._logger.info('Reading the outlet %s.' % outlet)
            outlet = WBW_ENV.read_vector(fspath(outlet))

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
        dem_preproc_mask_path = self.out_dir / 'dem_preproc_mask.tif'
        raster_masking_tool = SAGA_ENV / 'grid_tools' / 'Grid Masking'
        output = (
            raster_masking_tool.execute(
                grid=watershed_output.flow.dem_preproc.file_name,
                mask=watershed_output.watershed.file_name,
                masked=dem_preproc_mask_path,
            )
            .rasters['masked']
            .path
        )

        # ---- Computing the slope gradient for the masked DEM ----
        self._logger.info(
            'Computing the slope gradient for the streams in the watershed.'
        )
        dem_preproc_mask = WBW_ENV.read_raster(fspath(output))
        flow_watershed = self.compute_flow(dem_preproc_mask)
        streams = WBW_ENV.extract_streams(
            flow_watershed.flow_accumulation, streams_flow_accum_threshold
        )
        slope_gradient = degree_to_percent(
            WBW_ENV.stream_slope_continuous(
                flow_watershed.d8_pointer, streams, dem_preproc_mask
            )
        )
        slope_gradient_output = SlopeGradientComputationOutput(
            watershed_output, flow_watershed, streams, slope_gradient
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
        verbose=True,
        elevation=elevation,
        slope=out_slope,
        unit_slope=2,  # Percent rise
        method=6,  # 9 parameter 2nd order polynom (Zevenbergen & Thorne 1987)
    ).rasters['slope']


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
        verbose=True,
        dem=dem,
        values=rasters_str,
        lines=lines,
        profile=out_profile,
        name=split_by_field,
    ).vectors['profile']


def degree_to_percent(raster: WhiteboxRaster) -> WhiteboxRaster:
    return raster.to_radians().tan() * 100


class InterimData(Enum):
    DEM_30M_PREPROC = INTERIM_DATA_DIR / '30m_dem_preproc.tif'
    DEM_30M_D8_POINTER = INTERIM_DATA_DIR / '30m_d8_pointer.tif'
    DEM_30M_FLOW_ACCUM = INTERIM_DATA_DIR / '30m_flow_accumulation.tif'
    DEM_90M_PREPROC = INTERIM_DATA_DIR / '90m_dem_preproc.tif'
    DEM_90M_D8_POINTER = INTERIM_DATA_DIR / '90m_d8_pointer.tif'
    DEM_90M_FLOW_ACCUM = INTERIM_DATA_DIR / '90m_flow_accumulation.tif'

    def _get_dem_preproc(self, dem: Path) -> WhiteboxRaster:
        c_logger = logger.getChild(self.__class__.__name__)
        dem_preproc = HydrologicAnalysis(dem, INTERIM_DATA_DIR).preprocess_dem()
        write_whitebox(
            dem_preproc, self.value, overwrite=False, logger=c_logger
        )
        return dem_preproc

    def _get_flow_output(
        self, dem_preproc: WhiteboxRaster, prefix: str
    ) -> FlowAccumulationComputationOutput:
        c_logger = logger.getChild(self.__class__.__name__)
        hydro_analysis = HydrologicAnalysis(
            dem_preproc, out_dir=INTERIM_DATA_DIR
        )
        output = hydro_analysis.compute_flow(hydro_analysis.dem)
        output.write_whitebox(
            self.value.parent, prefix=prefix, overwrite=False, logger=c_logger
        )
        return output

    @t.overload
    def _get(self, *, as_whitebox: t.Literal[True]) -> WhiteboxRaster: ...
    @t.overload
    def _get(self, *, as_whitebox: t.Literal[False] = False) -> Path: ...
    def _get(self, *, as_whitebox: bool = False) -> Path | WhiteboxRaster:
        c_logger = logger.getChild(self.__class__.__name__)
        if self.value.exists():
            if as_whitebox:
                c_logger.info('Reading DEM at %s.' % self.value)
                return WBW_ENV.read_raster(fspath(self.value))
            return self.value
        match self:
            case InterimData.DEM_30M_PREPROC:
                ret = self._get_dem_preproc(DEM_30M)
            case InterimData.DEM_30M_D8_POINTER:
                ret = self._get_flow_output(
                    InterimData.DEM_30M_PREPROC._get(as_whitebox=True),
                    '30m_',
                ).d8_pointer
            case InterimData.DEM_30M_FLOW_ACCUM:
                ret = self._get_flow_output(
                    InterimData.DEM_30M_PREPROC._get(as_whitebox=True),
                    '30m_',
                ).flow_accumulation
            case InterimData.DEM_90M_PREPROC:
                ret = self._get_dem_preproc(DEM_90M)
            case InterimData.DEM_90M_D8_POINTER:
                ret = self._get_flow_output(
                    InterimData.DEM_90M_PREPROC._get(as_whitebox=True),
                    '90m_',
                ).d8_pointer
            case InterimData.DEM_90M_FLOW_ACCUM:
                ret = self._get_flow_output(
                    InterimData.DEM_90M_PREPROC._get(as_whitebox=True),
                    '90m_',
                ).flow_accumulation
        if not as_whitebox:
            return Path(ret.file_name)
        return ret


if __name__ == '__main__':
    gully_number = '2'
    # dem = RAW_DATA_DIR / 'ravene' / gully_number / '5 m' / 'merged.tif'
    outlet = RAW_DATA_DIR / 'ravene' / gully_number / 'pour_point.shp'
    dem = DATA_DIR / 'dem_30m.tif'
    hydro = HydrologicAnalysis(dem, out_dir=INTERIM_DATA_DIR / gully_number)
    watershed = hydro.compute_watershed(outlet, outlet_snap_dist=100)
    write_whitebox(
        WBW_ENV.raster_to_vector_polygons(watershed.watershed),
        INTERIM_DATA_DIR / gully_number / 'watershed.shp',
        overwrite=True,
    )
