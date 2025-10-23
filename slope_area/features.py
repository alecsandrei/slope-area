from __future__ import annotations

from collections import UserList
import collections.abc as c
import concurrent.futures
from dataclasses import dataclass, field
import logging
from os import makedirs
from pathlib import Path
import typing as t

import geopandas as gpd
import pyproj
import rasterio as rio
from rasterio import CRS
from rasterio.warp import Resampling
from rio_vrt import build_vrt
import shapely
from shapely.geometry.base import BaseGeometry
from whitebox_workflows import (
    AttributeField,
    FieldData,
    FieldDataType,
    Point2D,
    VectorGeometry,
    VectorGeometryType,
)
from whitebox_workflows.whitebox_workflows import Vector as WhiteboxVector

from slope_area import WBW_ENV
from slope_area._typing import Resolution
from slope_area.config import (
    DEM_DIR,
    DEM_DIR_EPSG,
    DEM_TILES,
)
from slope_area.geomorphometry import GeneralizedDEM
from slope_area.logger import create_logger
from slope_area.utils import (
    redirect_warnings,
    resample,
    timeit,
    write_whitebox,
)

if t.TYPE_CHECKING:
    from os import PathLike

    from whitebox_workflows.whitebox_workflows import WbEnvironment


logger = create_logger(__name__)


@dataclass
class VRT:
    path: Path
    dem_tiles: DEMTiles
    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self):
        self._logger = logger.getChild(self.__class__.__name__)

    def resample[T: PathLike](self, out_file: T, res: Resolution) -> T:
        crs = self.dem_tiles.gdf.crs.to_wkt()
        rio_crs = CRS.from_string(crs)
        reproject_kwargs = {
            'src_crs': rio_crs,
            'dst_crs': rio_crs,
            'resampling': Resampling.bilinear,
            'src_nodata': 0.0,
            'dst_nodata': -32767,
        }
        self._logger.info('Resampling VRT %s at res=%s' % (self.path, res))
        return resample(
            self.path, out_file, res=res, kwargs_reproject=reproject_kwargs
        )


class Feature(t.TypedDict):
    type: t.Literal['Feature']
    properties: c.Mapping[str, t.Any]
    geometry: BaseGeometry


class FeatureCollection(t.TypedDict):
    type: t.Literal['FeatureCollection']
    features: c.MutableSequence[Feature]


@dataclass
class DEMTilesBuilder:
    @classmethod
    def build(cls) -> None:
        c_logger = logger.getChild(cls.__name__)
        assert DEM_DIR.exists(), f'{DEM_DIR} does not exist'

        if DEM_TILES.exists():
            c_logger.info('Found DEM tiles at %s' % DEM_TILES)
            return None

        c_logger.info(
            'Extracting raster boundaries from %s to %s' % (DEM_DIR, DEM_TILES)
        )

        rasters = list(DEM_DIR.rglob('*.tif'))
        total = len(rasters)
        if not total:
            raise FileNotFoundError(f'No .tif rasters found in {DEM_DIR}')

        feature_coll: FeatureCollection = {
            'type': 'FeatureCollection',
            'features': [],
        }

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(cls.get_feature, raster) for raster in rasters
            ]

            for i, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                if not future.exception():
                    feature_coll['features'].append(future.result())
                if i % 1000 == 0 or i == total:
                    c_logger.info('Completed %i/%i rasters', i, total)

        gdf = gpd.GeoDataFrame.from_features(feature_coll, crs=DEM_DIR_EPSG)
        logger.info('Saving DEM boundaries to %s' % DEM_TILES)
        gdf.to_file(DEM_TILES)

    @classmethod
    def get_feature(cls, raster: Path) -> Feature:
        return {
            'type': 'Feature',
            'properties': {'path': raster.relative_to(DEM_DIR).as_posix()},
            'geometry': cls.get_raster_bounds(raster),
        }

    @staticmethod
    def get_raster_bounds(raster: Path) -> shapely.Polygon:
        try:
            with rio.open(raster) as src:
                return shapely.box(*src.bounds)
        except Exception as e:
            logger.error(
                'Failed to compute the boundary of the raster %s' % raster
            )
            raise e


@dataclass
class DEMTiles:
    gdf: gpd.GeoDataFrame
    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self):
        self._logger = logger.getChild(self.__class__.__name__)

    def build_vrt(self, out_vrt: PathLike) -> VRT:
        paths = [DEM_DIR / path for path in self.gdf['path']]
        self._logger.info('Building VRT from %i rasters.' % len(paths))
        return VRT(build_vrt(out_vrt, paths), self)

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

    @classmethod
    @timeit(logger, level=logging.INFO)
    def from_outlet(
        cls,
        outlet: Outlet,
        generalized_dem: GeneralizedDEM,
        out_dir: PathLike,
        outlet_snap_dist: int = 100,
        wbw_env: WbEnvironment = WBW_ENV,
    ) -> t.Self:
        return cls._from_outlet_watershed(
            outlet,
            generalized_dem=generalized_dem,
            out_dir=out_dir,
            outlet_snap_dist=outlet_snap_dist,
            wbw_env=wbw_env,
        )

    @classmethod
    def _from_outlet_watershed(
        cls,
        outlet: Outlet,
        generalized_dem: GeneralizedDEM,
        out_dir: PathLike,
        outlet_snap_dist: int = 100,
        wbw_env: WbEnvironment = WBW_ENV,
    ) -> t.Self:
        c_logger = logger.getChild(cls.__name__)
        c_logger.info('Infering DEM tiles based on the outlet.')
        out_dir = Path(out_dir)
        makedirs(out_dir, exist_ok=True)

        # ---- Read data ----
        wbw_outlet = Outlets([outlet], crs=outlet.crs).to_whitebox_vector()

        # ---- Computing watershed ----
        wbw_outlet_snapped = wbw_env.snap_pour_points(
            wbw_outlet,
            flow_accum=generalized_dem.flow_accum,
            snap_dist=outlet_snap_dist,
        )
        watershed = wbw_env.watershed(
            generalized_dem.d8_pointer, wbw_outlet_snapped
        )
        watershed_vec = wbw_env.raster_to_vector_polygons(watershed)
        write_whitebox(
            watershed_vec,
            out_dir / 'watershed.shp',
            logger=c_logger,
            overwrite=True,
        )
        with redirect_warnings(c_logger, RuntimeWarning, 'pyogrio.raw'):
            watershed = (
                gpd.read_file(watershed_vec.file_name)
                .geometry.make_valid()
                .iloc[0]
            )
        return cls.from_polygon(watershed)


@dataclass(frozen=True, eq=True)
class Outlet:
    geom: shapely.Point
    crs: pyproj.CRS
    name: str | None = None

    def __str__(self) -> str:
        return self.name or ''

    def __repr__(self) -> str:
        return f'Outlet(geom={self.geom}, crs={self.crs.to_epsg()}, name={self.name})'


@dataclass
class Outlets(UserList[Outlet]):
    def __init__(
        self, data: c.Iterable[Outlet] | None = None, *, crs: pyproj.CRS
    ):
        super().__init__(data)
        self.crs = crs

    @classmethod
    def from_gdf(
        cls, gdf: gpd.GeoDataFrame, name_field: str | None = None
    ) -> t.Self:
        if gdf.empty:
            raise ValueError('GeoDataFrame is empty — cannot create Outlets.')
        if not all(isinstance(geom, shapely.Point) for geom in gdf.geometry):
            raise TypeError('All geometries must be Points.')

        crs = pyproj.CRS.from_user_input(gdf.crs)
        outlets = [
            Outlet(
                geom=row.geometry,
                crs=crs,
                name=(row[name_field] if name_field else None),
            )
            for _, row in gdf.iterrows()
        ]
        return cls(outlets, crs=crs)

    def to_whitebox_vector(self) -> WhiteboxVector:
        vector = WBW_ENV.new_vector(
            VectorGeometryType.Point,
            [
                AttributeField(  # type: ignore
                    name='name',
                    field_type=FieldDataType.Text,
                    field_length=50,
                    decimal_count=0,
                )
            ],
            self.crs.to_wkt(),
        )
        for outlet in self.data:
            geometry = VectorGeometry.new_vector_geometry(
                VectorGeometryType.Point
            )
            point = Point2D(outlet.geom.x, outlet.geom.y)  # type: ignore
            geometry.add_point(point)
            vector.add_record(geometry)
            vector.add_attribute_record(
                rec=[
                    FieldData.new_text(outlet.name)
                    if outlet.name is not None
                    else FieldData.new_null()
                ],
                deleted=False,
            )
        return vector
