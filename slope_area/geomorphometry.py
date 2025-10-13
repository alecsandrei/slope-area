from __future__ import annotations

import collections.abc as c
from dataclasses import dataclass
from functools import partial
import logging
import math
from os import PathLike, fspath, makedirs
from pathlib import Path

from PySAGA_cmd import Raster as SAGARaster
from PySAGA_cmd import Vector as SAGAVector
from whitebox_workflows.whitebox_workflows import Raster as WhiteboxRaster
from whitebox_workflows.whitebox_workflows import Vector as WhiteboxVector

from slope_area import SAGA_ENV, WBW_ENV
from slope_area.config import (
    DATA_DIR,
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
    flow_watershed: FlowAccumulationComputationOutput
    streams_watershed: WhiteboxRaster
    slope_gradient_watershed: WhiteboxRaster


@dataclass
class HydrologicAnalysis:
    dem: WhiteboxRaster
    out_dir: Path

    def __init__(self, dem: PathLike, out_dir: PathLike = INTERIM_DATA_DIR):
        logger.info('Reading the DEM at %s.' % dem)
        self.dem = WBW_ENV.read_raster(fspath(dem))
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True)

    @timeit(logger, logging.DEBUG)
    def preprocess_dem(self) -> WhiteboxRaster:
        logger.info('Breaching depressions in the DEM.')
        dem_preproc = WBW_ENV.breach_depressions_least_cost(
            dem=self.dem, fill_deps=True
        )
        logger.info('Breaching single-cell pits.')
        dem_preproc = WBW_ENV.breach_single_cell_pits(dem_preproc)
        return dem_preproc

    @timeit(logger, logging.DEBUG)
    def compute_flow(
        self, dem_preproc: WhiteboxRaster | None = None
    ) -> FlowAccumulationComputationOutput:
        dem_preproc = (
            dem_preproc if dem_preproc is not None else self.preprocess_dem()
        )
        d8_pointer = WBW_ENV.d8_pointer(dem_preproc)
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
            outlet = WBW_ENV.read_vector(fspath(outlet))
        flow_output = self.compute_flow(dem_preproc)
        if outlet_snap_dist:
            outlet = WBW_ENV.snap_pour_points(
                outlet, flow_output.flow_accumulation, outlet_snap_dist
            )
        watershed = WBW_ENV.watershed(flow_output.d8_pointer, outlet)
        return WatershedComputationOutput(flow_output, outlet, watershed)

    @timeit(logger, logging.DEBUG)
    def compute_slope_gradient(
        self,
        outlet: WhiteboxVector | Path,
        streams_flow_accum_threshold: int = 100,
        outlet_snap_dist: int | None = None,
    ):
        write_whitebox_func = partial(
            write_whitebox, logger=logger, overwrite=True
        )
        if isinstance(outlet, Path):
            outlet = WBW_ENV.read_vector(outlet.as_posix())
        watershed_output = self.compute_watershed(outlet, outlet_snap_dist)
        write_whitebox_func(
            watershed_output.flow.dem_preproc,
            self.out_dir / 'dem_preproc.tif',
        )
        write_whitebox_func(
            watershed_output.watershed,
            self.out_dir / 'watershed.tif',
        )

        logger.info('Masking the preprocessed DEM with the watershed.')
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

        dem_preproc_mask = WBW_ENV.read_raster(fspath(output))
        flow_watershed = self.compute_flow(dem_preproc_mask)
        streams = WBW_ENV.extract_streams(
            flow_watershed.flow_accumulation, streams_flow_accum_threshold
        )
        slope_gradient = WBW_ENV.stream_slope_continuous(
            flow_watershed.d8_pointer, streams, dem_preproc_mask
        )
        slope_gradient_output = SlopeGradientComputationOutput(
            watershed_output, flow_watershed, streams, slope_gradient
        )
        write_whitebox_func(
            slope_gradient_output.slope_gradient_watershed,
            self.out_dir / 'slope_gradient.tif',
        )
        return slope_gradient_output


def compute_slope(elevation: PathLike, out_slope: PathLike) -> SAGARaster:
    tool = SAGA_ENV / 'ta_morphometry' / 'Slope, Aspect, Curvature'
    return tool.execute(
        verbose=True,
        elevation=elevation,
        slope=out_slope,
        unit_slope=2,  # Percent rise
        method=6,  # 9 parameter 2nd order polynom (Zevenbergen & Thorne 1987)
    ).rasters['slope']


def compute_profile_from_lines(
    rasters: c.Sequence[PathLike],
    lines: PathLike,
    out_profile: PathLike,
    dem: PathLike | None = None,
    split_by_field: str = 'FID',
) -> SAGAVector:
    if dem is None:
        dem = rasters[0]
    profiles_from_lines = SAGA_ENV / 'ta_profiles' / 'Profiles from Lines'
    return profiles_from_lines.execute(
        verbose=True,
        dem=dem,
        values=';'.join(fspath(raster) for raster in rasters),
        lines=lines,
        profile=out_profile,
        name=split_by_field,
    ).vectors['profile']


def degree_to_percent(degree: float) -> float:
    return math.tan(math.radians(degree)) * 100


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
