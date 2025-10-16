from __future__ import annotations

import collections.abc as c
import concurrent.futures
from dataclasses import dataclass
from pathlib import Path

from geopandas import gpd
import pandas as pd
from PySAGA_cmd import Raster as SAGARaster
from PySAGA_cmd import Vector as SAGAVector
from whitebox_workflows.whitebox_workflows import Vector as WhiteboxVector

from slope_area import SAGA_RASTER_SUFFIX, WBW_ENV
from slope_area._typing import Resolution
from slope_area.features import (
    VRT,
    DEMTiles,
    DEMTilesInferenceMethod,
    Outlet,
    Outlets,
)
from slope_area.geomorphometry import (
    HydrologicAnalysis,
    SlopeGradientComputationOutput,
    compute_profile_from_lines,
    compute_slope,
)
from slope_area.plot import preprocess_trial_results, slope_area_grid
from slope_area.utils import write_whitebox


@dataclass
class ResolutionPlotBuilder:
    outlet: Outlet
    resolutions: c.Sequence[Resolution]
    out_dir: Path

    def build(self):
        dem = self.out_dir / 'dem.vrt'
        vrt = DEMTiles.from_outlet(
            self.outlet,
            self.out_dir,
            method=DEMTilesInferenceMethod.WATERSHED,
            outlet_snap_dist=100,
        ).build_vrt(dem)
        trials = [
            Trial(
                self.outlet,
                self.out_dir / f'{resolution[0]}_{resolution[1]}',
                str(resolution),
                resolution,
                vrt=vrt,
            )
            for resolution in self.resolutions
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            results = executor.map(Trial.run, trials)
        slope_area_grid(
            preprocess_trial_results(results),
            'Test',
            'resolution',
            self.out_dir / 'figure.png',
        )


@dataclass
class Trial:
    outlet: Outlet
    out_dir: Path
    name: str
    resolution: Resolution
    vrt: VRT | None = None

    def __post_init__(self):
        self.out_dir.mkdir(exist_ok=True)
        if self.vrt is None:
            dem_tiles = DEMTiles.from_outlet(
                self.outlet,
                self.out_dir,
                method=DEMTilesInferenceMethod.WATERSHED,
            )
            self.vrt = dem_tiles.build_vrt(self.out_dir / 'dem.vrt')

    def get_resampled_dem(self) -> Path:
        dem_resampled_path = self.out_dir / 'dem_resampled.tif'
        assert self.vrt is not None
        return self.vrt.resample(
            out_file=dem_resampled_path,
            res=self.resolution,
        )

    def get_3x3_slope(self, dem: Path) -> SAGARaster:
        slope_3x3_path = (self.out_dir / 'slope').with_suffix(
            SAGA_RASTER_SUFFIX
        )
        return compute_slope(dem, slope_3x3_path)

    def get_slope_gradient(self, dem: Path) -> SlopeGradientComputationOutput:
        hydro_analysis = HydrologicAnalysis(dem, out_dir=self.out_dir)
        wbw_outlet = Outlets(
            [self.outlet], crs=self.outlet.crs
        ).to_whitebox_vector()
        return hydro_analysis.compute_slope_gradient(
            wbw_outlet,
            streams_flow_accum_threshold=100,
            outlet_snap_dist=100,
        )

    def get_streams(
        self, slope_grad: SlopeGradientComputationOutput
    ) -> WhiteboxVector:
        streams_vec_path = self.out_dir / 'main_streams.shp'
        streams_vec = WBW_ENV.raster_streams_to_vector(
            slope_grad.streams,
            slope_grad.flow.d8_pointer,
        )
        return write_whitebox(streams_vec, streams_vec_path, overwrite=True)

    def get_profiles(self) -> SAGAVector:
        profiles_path = self.out_dir / 'profiles.shp'
        dem = self.get_resampled_dem()
        slope_3x3 = self.get_3x3_slope(dem)
        slope_grad = self.get_slope_gradient(dem)
        streams = self.get_streams(slope_grad)

        return compute_profile_from_lines(
            dem=dem,
            rasters=[
                slope_grad.slope_grad.file_name,
                slope_grad.flow.flow_accumulation.file_name,
                slope_3x3.path,
            ],
            lines=streams.file_name,
            out_profile=profiles_path,
        )

    def run(self) -> TrialResult:
        return TrialResult(
            self.name, gpd.read_file(self.get_profiles()), self.resolution
        )


@dataclass
class TrialResult:
    name: str
    profiles: pd.DataFrame
    resolution: Resolution
