from __future__ import annotations

from slope_area import SAGA_RASTER_SUFFIX, WBW_ENV
from slope_area.config import (
    DATA_DIR,
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
)
from slope_area.features import DEMTiles
from slope_area.geomorphometry import (
    HydrologicAnalysis,
    compute_profile_from_lines,
    compute_slope,
)
from slope_area.utils import write_whitebox


def main(gully_number: str):
    out_dir = INTERIM_DATA_DIR / gully_number
    out_dir.mkdir(exist_ok=True)
    outlet = RAW_DATA_DIR / 'ravene' / gully_number / 'pour_point.shp'
    dem = DATA_DIR / 'dem_90m_breached.tif'
    vrt_path = out_dir / 'dem.vrt'
    resampled_path = out_dir / 'dem_resampled.tif'
    slope_3x3_path = (out_dir / 'slope').with_suffix(SAGA_RASTER_SUFFIX)
    streams_vec_path = out_dir / 'main_streams.shp'
    profile_path = out_dir / 'profile.shp'

    demtiles = DEMTiles.from_outlet(outlet, dem, basins_strahler_order=3)
    dem = demtiles.resample(
        demtiles.build_vrt(vrt_path),
        out_file=resampled_path,
        res=(5, 5),
    )

    hydro_analysis = HydrologicAnalysis(dem, out_dir=out_dir)
    slope_gradient = hydro_analysis.compute_slope_gradient(
        outlet, outlet_snap_dist=100
    )
    streams_vec = WBW_ENV.raster_streams_to_vector(
        slope_gradient.streams,
        slope_gradient.flow.d8_pointer,
    )
    write_whitebox(streams_vec, streams_vec_path, overwrite=True)

    slope_3x3 = compute_slope(dem, slope_3x3_path)

    profile = compute_profile_from_lines(
        dem=dem,
        rasters=[
            slope_gradient.slope_grad.file_name,
            slope_3x3.path,
        ],
        lines=streams_vec_path,
        out_profile=profile_path,
    )


if __name__ == '__main__':
    for number in range(1, 20):
        main(str(number))
