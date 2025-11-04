from __future__ import annotations

from collections import UserList
import collections.abc as c
from dataclasses import dataclass
from os import fspath
from pathlib import Path
import typing as t

import geopandas as gpd
import pyproj
import rasterio as rio
from rasterio.warp import Resampling
import shapely
from whitebox_workflows import (
    AttributeField,
    FieldData,
    FieldDataType,
    Point2D,
    VectorGeometry,
    VectorGeometryType,
)

from slope_area.config import get_wbw_env
from slope_area.logger import create_logger
from slope_area.utils import resample

if t.TYPE_CHECKING:
    from whitebox_workflows.whitebox_workflows import Vector as WhiteboxVector

    from slope_area._typing import AnyCRS, AnyLogger, Resolution, StrPath


m_logger = create_logger(__name__)


@dataclass(frozen=True, eq=True)
class Raster:
    path: StrPath

    def __fspath__(self) -> str:
        return fspath(self.path)

    @property
    def crs(self) -> rio.CRS:
        with rio.open(self.path) as src:
            return src.crs

    def define_projection(self, crs: AnyCRS) -> t.Self:
        if isinstance(crs, int):
            crs = pyproj.CRS.from_epsg(crs)
        with rio.open(self.path) as src:
            meta = src.meta.copy()
            meta.update(crs=crs)
            with rio.open(self.path, 'w', **meta) as out_src:
                out_src.write(src.read())
        return self

    def resample(
        self,
        out_file: StrPath,
        resolution: Resolution,
        crs: AnyCRS | None = None,
        *,
        logger: AnyLogger = m_logger,
    ) -> Raster:
        if crs is None:
            crs = self.crs
        reproject_kwargs = {
            'src_crs': crs,
            'dst_crs': crs,
            'resampling': Resampling.bilinear,
            'src_nodata': 0.0,
            'dst_nodata': -32767,
        }
        logger.info(
            'Resampling Raster %s at resolution=%s'
            % (Path(self.path).name, resolution)
        )
        return Raster(
            resample(
                self.path,
                out_file,
                res=resolution,
                kwargs_reproject=reproject_kwargs,
            )
        )


@dataclass(frozen=True, eq=True)
class Outlet:
    name: str
    geom: shapely.Point

    @classmethod
    def from_xy(cls, x: float, y: float, *, name: str) -> t.Self:
        return cls(name=name, geom=shapely.Point(x, y))


@dataclass
class Outlets(UserList[Outlet]):
    def __init__(self, data: c.Iterable[Outlet] | None = None):
        super().__init__(data)

    @classmethod
    def from_gdf(
        cls, gdf: gpd.GeoDataFrame, name_field: str | None = None
    ) -> t.Self:
        if gdf.empty:
            raise ValueError('GeoDataFrame is empty — cannot create Outlets')
        if not all(isinstance(geom, shapely.Point) for geom in gdf.geometry):
            raise TypeError('All geometries must be Points')

        outlets = [
            Outlet(
                name=(row[name_field] if name_field else str(i + 1)),
                geom=row.geometry,
            )
            for i, row in gdf.iterrows()
        ]
        return cls(outlets)

    def to_whitebox_vector(self, crs: AnyCRS) -> WhiteboxVector:
        if isinstance(crs, int):
            crs = pyproj.CRS.from_epsg(crs)
        vector = get_wbw_env().new_vector(
            VectorGeometryType.Point,
            [
                AttributeField(  # type: ignore
                    name='name',
                    field_type=FieldDataType.Text,
                    field_length=50,
                    decimal_count=0,
                )
            ],
            crs.to_wkt(),
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
