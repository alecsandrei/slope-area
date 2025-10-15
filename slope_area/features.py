from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from os import fspath
from pathlib import Path
import typing as t
import warnings

import geopandas as gpd
from rasterio import CRS
from rasterio.warp import Resampling
from rio_vrt import build_vrt
import shapely

from slope_area import WBW_ENV
from slope_area.config import (
    DATA_DIR,
    DEM_DIR,
    DEM_TILES,
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
)
from slope_area.geomorphometry import (
    InterimData,
)
from slope_area.logger import create_logger
from slope_area.utils import resample, timeit, write_whitebox

if t.TYPE_CHECKING:
    from os import PathLike

logger = create_logger(__name__)


class DEMTilesInferenceMethod(Enum):
    STRAHLER_BASINS = auto()
    WATERSHED = auto()


@dataclass
class DEMTiles:
    gdf: gpd.GeoDataFrame
    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self):
        self._logger = logger.getChild(self.__class__.__name__)

    def build_vrt(self, out_vrt: PathLike) -> Path:
        paths = [DEM_DIR / path for path in self.gdf['path']]
        self._logger.info('Building VRT from %i rasters.' % len(paths))
        return build_vrt(out_vrt, paths)

    def resample[T: PathLike](
        self, vrt: PathLike, out_file: T, res: tuple[float, float]
    ) -> T:
        crs = self.gdf.crs.to_wkt()
        rio_crs = CRS.from_string(crs)
        # NOTE: nodata=0 might be buggy e.g. when 0 is values are present in DEM
        reproject_kwargs = {
            'src_crs': rio_crs,
            'dst_crs': rio_crs,
            'resampling': Resampling.bilinear,
            'src_nodata': 0.0,
            'dst_nodata': 0.0,
        }
        self._logger.info('Resampling VRT %s at res=%s' % (vrt, res))
        return resample(
            vrt, out_file, res=res, kwargs_reproject=reproject_kwargs
        )

    @classmethod
    def from_polygon(
        cls, polygon: shapely.Polygon | shapely.MultiPolygon
    ) -> t.Self:
        c_logger = logger.getChild(cls.__name__)
        dem_tiles = gpd.read_file(DEM_TILES)
        subset = dem_tiles[dem_tiles.intersects(polygon)]
        c_logger.info(
            'Extracted %i tiles based on the polygon %r'
            % (subset.shape[0], polygon)
        )
        return cls(subset)

    @t.overload
    @classmethod
    def from_outlet(
        cls,
        outlet: PathLike,
        out_dir: PathLike,
        method: t.Literal[
            DEMTilesInferenceMethod.STRAHLER_BASINS
        ] = DEMTilesInferenceMethod.STRAHLER_BASINS,
        outlet_snap_dist: int = 100,
        stream_units_threshold: int = 100,
        basins_strahler_order: int = 5,
    ) -> t.Self: ...
    @t.overload
    @classmethod
    def from_outlet(
        cls,
        outlet: PathLike,
        out_dir: PathLike,
        method: t.Literal[
            DEMTilesInferenceMethod.WATERSHED
        ] = DEMTilesInferenceMethod.WATERSHED,
        outlet_snap_dist: int = 100,
    ) -> t.Self: ...
    @classmethod
    @timeit(logger, level=logging.INFO)
    def from_outlet(
        cls,
        outlet: PathLike,
        out_dir: PathLike,
        method: DEMTilesInferenceMethod = DEMTilesInferenceMethod.WATERSHED,
        outlet_snap_dist: int = 100,
        stream_units_threshold: int = 100,
        basins_strahler_order: int = 5,
    ) -> t.Self:
        match method:
            case DEMTilesInferenceMethod.STRAHLER_BASINS:
                return cls._from_outlet_strahler_basins(
                    outlet,
                    out_dir=out_dir,
                    outlet_snap_dist=outlet_snap_dist,
                    stream_units_threshold=stream_units_threshold,
                    basins_strahler_order=basins_strahler_order,
                )
            case DEMTilesInferenceMethod.WATERSHED:
                return cls._from_outlet_watershed(
                    outlet,
                    out_dir=out_dir,
                    outlet_snap_dist=outlet_snap_dist,
                )

    @classmethod
    @timeit(logger, level=logging.INFO)
    def _from_outlet_watershed(
        cls,
        outlet: PathLike,
        out_dir: PathLike,
        outlet_snap_dist: int = 100,
    ) -> t.Self:
        c_logger = logger.getChild(cls.__name__)
        c_logger.info('Infering DEM tiles based on the outlet.')
        out_dir = Path(out_dir)

        ## ---- Read data ----
        d8_pointer = InterimData.DEM_30M_D8_POINTER._get(as_whitebox=True)
        flow_accum = InterimData.DEM_30M_FLOW_ACCUM._get(as_whitebox=True)

        # ---- Computing watershed ----
        c_logger.info('Reading outlet %s.' % outlet)
        wbw_outlet = WBW_ENV.snap_pour_points(
            WBW_ENV.read_vector(fspath(outlet)),
            flow_accum=flow_accum,
            snap_dist=outlet_snap_dist,
        )
        watershed = WBW_ENV.watershed(d8_pointer, wbw_outlet)
        watershed_vec = WBW_ENV.raster_to_vector_polygons(watershed)
        write_whitebox(
            watershed_vec,
            out_dir / 'watershed.shp',
            logger=c_logger,
            overwrite=True,
        )
        watershed = (
            gpd.read_file(watershed_vec.file_name).geometry.make_valid().iloc[0]
        )
        return cls.from_polygon(watershed)

    @classmethod
    @timeit(logger, level=logging.INFO)
    def _from_outlet_strahler_basins(
        cls,
        outlet: PathLike,
        out_dir: PathLike,
        outlet_snap_dist: int = 100,
        stream_units_threshold: int = 5,
        basins_strahler_order: int = 5,
    ) -> t.Self:
        c_logger = logger.getChild(cls.__name__)
        c_logger.info('Infering DEM tiles based on the outlet.')
        out_dir = Path(out_dir)

        # ---- Read data ----
        dem = InterimData.DEM_30M_PREPROC._get(as_whitebox=True)
        d8_pointer = InterimData.DEM_30M_D8_POINTER._get(as_whitebox=True)
        outlet_gdf = gpd.read_file(outlet)
        assert outlet_gdf.shape[0] == 1, 'Expected a single outlet'
        outlet_geom = outlet_gdf.geometry.iloc[0]

        # ---- Extract the large basin overlapping outlet ----
        basins_file = out_dir / 'basins.shp'
        basin_file = basins_file.with_stem('basin')
        c_logger.info(
            'Extracting the large basin intersecting with the outlet.'
        )
        basins = WBW_ENV.basins(d8_pointer)
        basins_as_vec = WBW_ENV.raster_to_vector_polygons(basins)
        WBW_ENV.write_vector(basins_as_vec, fspath(basins_file))
        with warnings.catch_warnings():
            # This gives RuntimeWarning about the geometry being invalid and corrected
            warnings.simplefilter('ignore')
            basins_gdf = gpd.read_file(basins_file)
        basins_gdf[basins_gdf.intersects(outlet_geom)].make_valid().to_file(
            basin_file
        )
        basin = WBW_ENV.read_vector(fspath(basin_file))

        # ---- Derive D8 pointer and flowacc from the masked DEM ----
        c_logger.info(
            'Masking the DEM with the large basin and generating D8 and flowacc.'
        )
        dem_mask = WBW_ENV.clip_raster_to_polygon(dem, basin)
        d8_pointer_mask = WBW_ENV.d8_pointer(dem_mask)
        flow_mask = WBW_ENV.d8_flow_accum(
            d8_pointer_mask, out_type='cells', input_is_pointer=True
        )

        # ---- Extract the subbasins based on the strahler threshold ----
        c_logger.info(
            'Delineating basins based on the threshold strahler order of %i.'
            % basins_strahler_order
        )
        streams = WBW_ENV.extract_streams(
            flow_mask, threshold=stream_units_threshold
        )
        streams_strahler = WBW_ENV.strahler_stream_order(
            d8_pointer_mask, streams
        )
        reclass: list[list[float]] = [
            [0, 0, basins_strahler_order],
            [1, basins_strahler_order, streams_strahler.configs.maximum],
        ]
        streams_strahler_threshold = WBW_ENV.reclass(streams_strahler, reclass)
        basins_strahler_threshold = WBW_ENV.subbasins(
            d8_pointer_mask, streams_strahler_threshold
        )
        basins_strahler_as_vec = WBW_ENV.raster_to_vector_polygons(
            basins_strahler_threshold
        )
        basins_strahler_as_vec_file = basins_file.with_stem('basins_strahler')
        WBW_ENV.write_vector(
            basins_strahler_as_vec, fspath(basins_strahler_as_vec_file)
        )

        # ---- Extract the basin(s) overlapping the outlet ----
        c_logger.info('Extracting the subbasin(s) overlapping the outlet.')
        with warnings.catch_warnings():
            # This gives RuntimeWarning about the geometry being invalid and corrected
            warnings.simplefilter('ignore')
            basins_strahler_gdf = gpd.read_file(basins_strahler_as_vec_file)
        basin_strahler: shapely.Polygon | shapely.MultiPolygon = (
            basins_strahler_gdf[
                basins_strahler_gdf.intersects(outlet_geom)
            ].union_all()
        )

        return cls.from_polygon(basin_strahler)


if __name__ == '__main__':
    gully_number = '2'
    outlet = RAW_DATA_DIR / 'ravene' / gully_number / 'pour_point.shp'
    dem = DATA_DIR / 'dem_90m_breached.tif'
    demtiles = DEMTiles.from_outlet(outlet, dem)
    demtiles.resample(
        demtiles.build_vrt(INTERIM_DATA_DIR / 'dem.vrt'),
        out_file=INTERIM_DATA_DIR / 'dem_resampled.tif',
        res=(5, 5),
    )
