from __future__ import annotations

import collections.abc as c
import concurrent.futures
from dataclasses import InitVar, dataclass, field
from functools import cache, partial
import logging
from os import fspath, makedirs
from pathlib import Path
import typing as t
import xml.etree.ElementTree as ET

import geopandas as gpd
import pyproj
import rasterio as rio
from rio_vrt import build_vrt
import shapely
from shapely.geometry.base import BaseGeometry
from whitebox_workflows.whitebox_workflows import Raster as WhiteboxRaster

from slope_area._typing import AnyCRS, DEMProvider
from slope_area.config import get_wbw_env
from slope_area.features import Outlet, Outlets, Raster
from slope_area.geomorphometry import (
    FlowAccumulationComputationOutput,
    compute_flow,
)
from slope_area.logger import create_logger
from slope_area.utils import redirect_warnings, timeit, write_whitebox

if t.TYPE_CHECKING:
    from whitebox_workflows.whitebox_workflows import WbEnvironment

    from slope_area._typing import AnyLogger, StrPath


m_logger = create_logger(__name__)


@dataclass(frozen=True)
class DEMSource:
    dem_dir: Path
    tiles: Path
    generalized_dem: GeneralizedDEM
    crs: AnyCRS


@dataclass(frozen=True, eq=True)
class VRT(Raster):
    dem_tiles: DEMTiles

    @t.override
    def define_projection(self, crs: AnyCRS) -> t.Self:
        if isinstance(crs, int):
            crs = pyproj.CRS.from_epsg(crs)
        crs_str = crs.to_wkt()

        # Read and parse XML
        tree = ET.parse(self.path)
        root = tree.getroot()

        # Find the SRS element and replace its text
        srs_elem = root.find('SRS')
        if srs_elem is None:
            srs_elem = ET.SubElement(root, 'SRS')
        srs_elem.text = crs_str

        # Save back to file
        tree.write(self.path, encoding='UTF-8')
        return self

    @classmethod
    def from_dem_tiles(
        cls, dem_tiles: DEMTiles, out_vrt: Path, *, logger: AnyLogger = m_logger
    ) -> t.Self:
        paths = [
            Path(dem_tiles.dem_dir) / path for path in dem_tiles.gdf['path']
        ]
        logger.info('Building VRT from %i rasters' % len(paths))
        return cls(build_vrt(out_vrt, paths), dem_tiles)


@dataclass(frozen=True, eq=True)
class GeneralizedDEM(Raster):
    out_dir: StrPath

    @property
    def dem_preproc(self) -> WhiteboxRaster:
        return self.get_flow_output().dem_preproc

    @property
    def d8_pointer(self) -> WhiteboxRaster:
        return self.get_flow_output().d8_pointer

    @property
    def flow_accum(self) -> WhiteboxRaster:
        return self.get_flow_output().flow_accumulation

    def read_rasters(
        self, rasters: c.Iterable[StrPath]
    ) -> list[WhiteboxRaster]:
        return [get_wbw_env().read_raster(fspath(raster)) for raster in rasters]

    def compute_flow(
        self, rasters: tuple[Path, Path, Path]
    ) -> FlowAccumulationComputationOutput:
        logger = m_logger.getChild(self.__class__.__name__)
        flow = compute_flow(self.path, logger=logger)
        outputs = (flow.dem_preproc, flow.d8_pointer, flow.flow_accumulation)
        write_whitebox_func = partial(
            write_whitebox,
            overwrite=True,
            logger=logger,
        )
        for wbw_raster, out_file in zip(outputs, rasters):
            write_whitebox_func(wbw_raster, out_file)
        return flow

    def read_flow_output(
        self, rasters: tuple[Path, Path, Path]
    ) -> FlowAccumulationComputationOutput:
        wbw_rasters = self.read_rasters(rasters)
        return FlowAccumulationComputationOutput(*wbw_rasters)

    def get_rasters(self, prefix: str) -> tuple[Path, Path, Path]:
        out_dir = Path(self.out_dir)
        return (
            out_dir / f'{prefix}dem_preproc.tif',
            out_dir / f'{prefix}d8_pointer.tif',
            out_dir / f'{prefix}flow_accumulation.tif',
        )

    @cache
    def get_flow_output(self) -> FlowAccumulationComputationOutput:
        prefix = Path(self.path).stem + '_'
        rasters = self.get_rasters(prefix)
        if all(raster.exists() for raster in rasters):
            m_logger.info(
                'Found rasters %s'
                % ', '.join([Path(raster).name for raster in rasters])
            )
            return self.read_flow_output(rasters)
        else:
            m_logger.info(
                'Computing rasters %s'
                % ', '.join([Path(raster).stem for raster in rasters])
            )
            return self.compute_flow(rasters)


class Feature(t.TypedDict):
    type: t.Literal['Feature']
    properties: c.Mapping[str, t.Any]
    geometry: BaseGeometry


class FeatureCollection(t.TypedDict):
    type: t.Literal['FeatureCollection']
    features: c.MutableSequence[Feature]


@dataclass
class DEMTilesBuilder:
    dem_dir: StrPath
    dem_dir_epsg: int
    tiles: StrPath
    _logger: logging.Logger = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger = m_logger.getChild(self.__class__.__name__)

    def build(self) -> Path:
        tiles = Path(self.tiles)
        if tiles.exists():
            self._logger.info('Found DEM tiles at %s' % self.tiles)
            return tiles

        self._logger.info(
            'Extracting raster boundaries from %s to %s'
            % (self.dem_dir, self.tiles)
        )

        rasters = list(Path(self.dem_dir).rglob('*.tif'))
        total = len(rasters)
        if not total:
            raise FileNotFoundError(f'No .tif rasters found in {self.dem_dir}')

        feature_coll: FeatureCollection = {
            'type': 'FeatureCollection',
            'features': [],
        }

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.get_feature, raster) for raster in rasters
            ]

            for i, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                if not future.exception():
                    feature_coll['features'].append(future.result())
                if i % 1000 == 0 or i == total:
                    self._logger.info('Completed %i/%i rasters', i, total)

        gdf = gpd.GeoDataFrame.from_features(
            feature_coll, crs=self.dem_dir_epsg
        )
        self._logger.info('Saving DEM boundaries to %s' % self.tiles)
        gdf.to_file(self.tiles)
        return tiles

    def get_feature(self, raster: Path) -> Feature:
        return {
            'type': 'Feature',
            'properties': {'path': raster.relative_to(self.dem_dir).as_posix()},
            'geometry': self.get_raster_bounds(raster),
        }

    def get_raster_bounds(self, raster: Path) -> shapely.Polygon:
        try:
            with rio.open(raster) as src:
                return shapely.box(*src.bounds)
        except Exception as e:
            self._logger.error(
                'Failed to compute the boundary of the raster %s' % raster
            )
            raise e


@dataclass
class DEMTiles:
    dem_dir: StrPath
    gdf: gpd.GeoDataFrame = field(repr=False)
    logger: InitVar[AnyLogger | None] = field(kw_only=True, default=None)
    _logger: AnyLogger = field(init=False, repr=False)

    def __post_init__(self, logger: AnyLogger | None) -> None:
        self._logger = logger or m_logger.getChild(self.__class__.__name__)

    @classmethod
    def from_polygon(
        cls,
        dem_dir: StrPath,
        tiles: StrPath,
        polygon: shapely.Polygon | shapely.MultiPolygon,
        *,
        logger: AnyLogger | None = None,
    ) -> t.Self:
        c_logger = m_logger.getChild(cls.__name__)
        dem_tiles = gpd.read_file(tiles)
        subset = dem_tiles[dem_tiles.intersects(polygon)]
        c_logger.info(
            'Extracted %i tiles based on the polygon %r'
            % (subset.shape[0], polygon)
        )
        return cls(dem_dir, subset, logger=logger)

    @classmethod
    @timeit(m_logger, level=logging.INFO)
    def from_outlet(
        cls,
        dem_source: DEMSource,
        outlet: Outlet,
        out_dir: StrPath,
        outlet_snap_dist: float = 100,
        wbw_env: WbEnvironment | None = None,
        *,
        logger: AnyLogger | None = None,
    ) -> t.Self:
        logger = logger or m_logger.getChild(cls.__name__)
        if wbw_env is None:
            wbw_env = get_wbw_env()
        logger.info('Infering DEM tiles based on the outlet')
        out_dir = Path(out_dir)
        makedirs(out_dir, exist_ok=True)

        # ---- Read data ----
        wbw_outlet = Outlets([outlet]).to_whitebox_vector(dem_source.crs)

        # ---- Computing watershed ----
        wbw_outlet_snapped = wbw_env.snap_pour_points(
            wbw_outlet,
            flow_accum=dem_source.generalized_dem.flow_accum,
            snap_dist=outlet_snap_dist,
        )
        watershed = wbw_env.watershed(
            dem_source.generalized_dem.d8_pointer, wbw_outlet_snapped
        )
        watershed_vec = wbw_env.raster_to_vector_polygons(watershed)
        write_whitebox(
            watershed_vec,
            out_dir / 'watershed.shp',
            logger=logger,
            overwrite=True,
        )
        with redirect_warnings(logger, RuntimeWarning, 'pyogrio.raw'):
            watershed = (
                gpd.read_file(watershed_vec.file_name)
                .geometry.make_valid()
                .iloc[0]
            )
        return cls.from_polygon(
            dem_source.dem_dir, dem_source.tiles, watershed, logger=logger
        )


@dataclass
class DynamicVRT(DEMProvider):
    dem_source: DEMSource
    out_parent: StrPath
    outlet_snap_distance: float

    def get_dem(
        self,
        outlet: Outlet,
        *,
        logger: AnyLogger | None = None,
    ) -> VRT:
        if logger is None:
            logger = m_logger.getChild(self.__class__.__name__)
        out_dir = Path(self.out_parent) / outlet.name
        makedirs(out_dir, exist_ok=True)
        dem_tiles = DEMTiles.from_outlet(
            dem_source=self.dem_source,
            outlet=outlet,
            out_dir=Path(self.out_parent) / outlet.name,
            outlet_snap_dist=self.outlet_snap_distance,
            wbw_env=get_wbw_env(),
            logger=logger,
        )
        vrt = VRT.from_dem_tiles(dem_tiles, out_dir / 'dem.vrt')
        if not vrt.crs.is_projected:
            vrt = vrt.define_projection(self.dem_source.crs)
        return vrt
