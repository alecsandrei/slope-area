from __future__ import annotations

import concurrent
from dataclasses import dataclass
from pathlib import Path

from geopandas import gpd
from PySAGA_cmd import Raster as SAGARaster
from PySAGA_cmd import Vector as SAGAVector
from whitebox_workflows.whitebox_workflows import Vector as WhiteboxVector

from slope_area import SAGA_RASTER_SUFFIX, WBW_ENV
from slope_area.config import (
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
)
from slope_area.features import DEMTiles, DEMTilesInferenceMethod
from slope_area.geomorphometry import (
    HydrologicAnalysis,
    SlopeGradientComputationOutput,
    compute_profile_from_lines,
    compute_slope,
)
from slope_area.utils import write_whitebox


@dataclass
class Trial:
    outlet: Path
    out_dir: Path
    resolution: tuple[float, float]

    def get_resampled_dem(self) -> Path:
        vrt_path = self.out_dir / 'dem.vrt'
        dem_resampled_path = self.out_dir / 'dem_resampled.tif'
        demtiles = DEMTiles.from_outlet(
            self.outlet,
            out_dir=self.out_dir,
            method=DEMTilesInferenceMethod.WATERSHED,
        )
        return demtiles.resample(
            demtiles.build_vrt(vrt_path),
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
        return hydro_analysis.compute_slope_gradient(
            self.outlet, streams_flow_accum_threshold=100, outlet_snap_dist=100
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

    def run(self):
        profiles = gpd.read_file(self.get_profiles())


def main(gully_number: str):
    out_dir = INTERIM_DATA_DIR / gully_number
    out_dir.mkdir(exist_ok=True)
    outlet = RAW_DATA_DIR / 'ravene' / gully_number / 'pour_point.shp'
    trial = Trial(outlet, out_dir, (5, 5)).run()


if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(main, map(str, range(1, 21)))
    breakpoint()
    # for number in range(1, 26): main(str(number))
