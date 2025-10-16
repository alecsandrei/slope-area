from __future__ import annotations

from collections import UserList
import collections.abc as c
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from os import fspath
from pathlib import Path
import typing as t
import warnings

import geopandas as gpd
from geopandas import gpd
import pyproj
from rasterio import CRS
from rasterio.warp import Resampling
from rio_vrt import build_vrt
import shapely
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
    DEM_TILES,
)
from slope_area.geomorphometry import InterimData
from slope_area.logger import create_logger
from slope_area.utils import (
    resample,
    timeit,
    write_whitebox,
)

if t.TYPE_CHECKING:
    from os import PathLike


logger = create_logger(__name__)


class DEMTilesInferenceMethod(Enum):
    STRAHLER_BASINS = auto()
    WATERSHED = auto()


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

    @t.overload
    @classmethod
    def from_outlet(
        cls,
        outlet: Outlet,
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
        outlet: Outlet,
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
        outlet: Outlet,
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
        outlet: Outlet,
        out_dir: PathLike,
        outlet_snap_dist: int = 100,
    ) -> t.Self:
        c_logger = logger.getChild(cls.__name__)
        c_logger.info('Infering DEM tiles based on the outlet.')
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)

        ## ---- Read data ----
        d8_pointer = InterimData.DEM_30M_D8_POINTER._get(as_whitebox=True)
        flow_accum = InterimData.DEM_30M_FLOW_ACCUM._get(as_whitebox=True)
        wbw_outlet = Outlets([outlet], crs=outlet.crs).to_whitebox_vector()

        # ---- Computing watershed ----
        wbw_outlet_snapped = WBW_ENV.snap_pour_points(
            wbw_outlet,
            flow_accum=flow_accum,
            snap_dist=outlet_snap_dist,
        )
        watershed = WBW_ENV.watershed(d8_pointer, wbw_outlet_snapped)
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
        outlet: Outlet,
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

        # ---- CRS check ----
        for raster in (dem, d8_pointer):
            if pyproj.CRS.from_string(raster.configs.projection) != outlet.crs:
                logger.error(
                    'CRS of outlet and raster %s do not match.'
                    % raster.file_name
                )

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
        basins_gdf[basins_gdf.intersects(outlet.geom)].make_valid().to_file(
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
                basins_strahler_gdf.intersects(outlet.geom)
            ].union_all()
        )

        return cls.from_polygon(basin_strahler)


@dataclass(frozen=True)
class Outlet:
    geom: shapely.Point
    crs: pyproj.CRS
    name: str | None = None


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
            if hasattr(Point2D, 'new'):
                point = Point2D.new(outlet.geom.x, outlet.geom.y)
            else:
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
