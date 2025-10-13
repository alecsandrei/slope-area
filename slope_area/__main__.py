from __future__ import annotations

from slope_area.config import DATA_DIR, INTERIM_DATA_DIR, RAW_DATA_DIR
from slope_area.features import DEMTiles
from slope_area.hydro import HydrologicAnalysis


def main():
    gully_number = '2'
    outlet = RAW_DATA_DIR / 'ravene' / gully_number / 'pour_point.shp'
    dem = DATA_DIR / 'dem_90m_breached.tif'
    demtiles = DEMTiles.from_outlet(outlet, dem)
    vrt_path = INTERIM_DATA_DIR / 'dem.vrt'
    resampled_path = INTERIM_DATA_DIR / 'dem_resampled.tif'
    dem = demtiles.resample(
        demtiles.build_vrt(vrt_path),
        out_file=resampled_path,
        res=(5, 5),
    )

    hydro_analysis = HydrologicAnalysis(dem)
    hydro_analysis.compute_slope_gradient(outlet, outlet_snap_dist=100)


if __name__ == '__main__':
    main()
