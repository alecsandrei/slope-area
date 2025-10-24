from __future__ import annotations

from abc import ABC, abstractmethod
import collections.abc as c
import concurrent.futures
from dataclasses import dataclass, field
from functools import cached_property
import logging
from os import fspath, makedirs
from pathlib import Path
import time

from geopandas import gpd
import pandas as pd
from PySAGA_cmd import Raster as SAGARaster
from rich import box
from rich.live import Live
from rich.table import Table
from whitebox_workflows import WbEnvironment
from whitebox_workflows.whitebox_workflows import Vector as WhiteboxVector

from slope_area import SAGA_RASTER_SUFFIX, get_wbw_env
from slope_area._typing import Resolution
from slope_area.config import (
    OUTLET_SNAP_DIST,
    STREAM_FLOW_ACCUM_THRESHOLD,
    TRIAL_WORKERS,
)
from slope_area.features import (
    VRT,
    DEMTiles,
    Outlet,
    Outlets,
)
from slope_area.geomorphometry import (
    GeneralizedDEM,
    HydrologicAnalysis,
    SlopeGradientComputationOutput,
    compute_slope,
)
from slope_area.logger import (
    MultiprocessingLog,
    RichDictHandler,
    create_logger,
    silent_logs,
)
from slope_area.plot import preprocess_trial_results, slope_area_grid
from slope_area.utils import redirect_warnings, write_whitebox

logger = create_logger(__name__)


def make_table(logs: dict[str, str]) -> Table:
    table = Table(box=box.SQUARE_DOUBLE_HEAD, show_header=True, expand=True)
    table.add_column('Trial')
    table.add_column('Status')
    for trial, log in logs.items():
        table.add_row(f'[bold]Trial "{trial}"[/bold]', log)
    return table


def create_rich_logger(
    logger_name: str, logs: dict[str, str]
) -> logging.Logger:
    r_logger = logger.getChild(logger_name)
    rich_handler = RichDictHandler(logs)
    handler = MultiprocessingLog([rich_handler])
    r_logger.addHandler(handler)
    return r_logger


@dataclass
class Builder(ABC):
    @abstractmethod
    def get_trials(self, logger: logging.Logger) -> list[Trial]: ...
    @abstractmethod
    def save_plot(self, trial_results: list[TrialResult]): ...
    @property
    @abstractmethod
    def trial_names(self) -> list[str]: ...

    def run_trials(
        self, trials: c.Iterable[Trial], logs: dict[str, str]
    ) -> list[concurrent.futures.Future[TrialResult]]:
        # Silence logs. Otherwise stdout would be filled with other messages
        with silent_logs('slopeArea'):
            with Live(
                make_table(logs),
                refresh_per_second=10,
                redirect_stderr=True,
                redirect_stdout=True,
            ) as live:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=TRIAL_WORKERS
                ) as executor:
                    futures = [
                        executor.submit(Trial.run, trial) for trial in trials
                    ]
                    while any(f.running() for f in futures):
                        live.update(make_table(logs))
                        time.sleep(0.1)
                    live.update(make_table(logs))
        return futures

    def build(self):
        logs = {name: '[dim]Waiting...[/dim]' for name in self.trial_names}
        m_logger = create_rich_logger(self.__class__.__name__, logs)
        trials = self.get_trials(m_logger)
        futures = self.run_trials(trials, logs)
        trial_results = [
            future.result() for future in futures if future.exception() is None
        ]
        self.save_plot(trial_results)


@dataclass
class ResolutionPlotBuilder(Builder):
    outlet: Outlet
    resolutions: c.Sequence[Resolution]
    generalized_dem: GeneralizedDEM
    out_dir: Path

    def get_vrt(self) -> VRT:
        dem = self.out_dir / 'dem.vrt'
        return DEMTiles.from_outlet(
            outlet=self.outlet,
            generalized_dem=self.generalized_dem,
            out_dir=self.out_dir,
            outlet_snap_dist=OUTLET_SNAP_DIST,
        ).build_vrt(dem)

    @cached_property
    def trial_names(self) -> list[str]:
        unit_name = self.outlet.crs.axis_info[0].unit_name
        return [
            f'{resolution[0]} {unit_name}' for resolution in self.resolutions
        ]

    def get_trials(self, logger: logging.Logger) -> list[Trial]:
        vrt = self.get_vrt()
        return [
            Trial(
                outlet=self.outlet,
                out_dir=self.out_dir / trial_name,
                name=trial_name,
                resolution=resolution,
                vrt=vrt,
                logger=logger,
            )
            for trial_name, resolution in zip(
                self.trial_names, self.resolutions
            )
        ]

    def save_plot(self, trial_results: c.Iterable[TrialResult]) -> None:
        slope_area_grid(
            data=preprocess_trial_results(trial_results),
            col='trial_name',
            out_fig=self.out_dir / 'slope_area.png',
        )


@dataclass
class OutletPlotBuilder(Builder):
    outlets: Outlets
    resolution: Resolution
    generalized_dem: GeneralizedDEM
    out_dir: Path

    @cached_property
    def trial_names(self) -> list[str]:
        return [str(outlet) for outlet in self.outlets]

    def get_trials(self, logger: logging.Logger) -> list[Trial]:
        return [
            Trial(
                outlet=outlet,
                out_dir=self.out_dir / trial_name,
                name=trial_name,
                resolution=self.resolution,
                generalized_dem=self.generalized_dem,
                logger=logger,
            )
            for trial_name, outlet in zip(self.trial_names, self.outlets)
        ]

    def save_plot(self, trial_results: c.Iterable[TrialResult]) -> None:
        slope_area_grid(
            data=preprocess_trial_results(trial_results),
            col='trial_name',
            out_fig=self.out_dir / 'slope_area.png',
        )


@dataclass
class Trial:
    outlet: Outlet
    out_dir: Path
    name: str
    resolution: Resolution
    generalized_dem: GeneralizedDEM | None = None
    vrt: VRT | None = None
    logger: logging.Logger | None = field(default=None, repr=False)
    logger_adapter: logging.LoggerAdapter = field(init=False, repr=False)
    _wbw_env: WbEnvironment | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        makedirs(self.out_dir, exist_ok=True)
        if self.logger is None:
            self.logger = create_logger(__name__)
        self.logger_adapter = logging.LoggerAdapter(
            self.logger, extra={'trialName': self.name}
        )

    def get_vrt(self) -> VRT:
        if self.vrt is None:
            if self.generalized_dem is None:
                raise ValueError(
                    'Please initialize Trial with either vrt or generalized_dem'
                )
            assert self.logger_adapter is not None
            self.logger_adapter.info('Generating VRT from DEM tiles')
            dem_tiles = DEMTiles.from_outlet(
                outlet=self.outlet,
                generalized_dem=self.generalized_dem,
                out_dir=self.out_dir,
            )
            self.vrt = dem_tiles.build_vrt(self.out_dir / 'dem.vrt')
        return self.vrt

    @property
    def wbw_env(self) -> WbEnvironment:
        if self._wbw_env is None:
            self._wbw_env = get_wbw_env()
        return self._wbw_env

    def get_resampled_dem(self, vrt: VRT) -> Path:
        dem_resampled_path = self.out_dir / 'dem_resampled.tif'
        vrt = self.get_vrt()
        return vrt.resample(
            out_file=dem_resampled_path,
            res=self.resolution,
        )

    def get_3x3_slope(self, dem: Path) -> SAGARaster:
        slope_3x3_path = (self.out_dir / 'slope').with_suffix(
            SAGA_RASTER_SUFFIX
        )
        return compute_slope(dem, slope_3x3_path)

    def get_slope_gradient(self, dem: Path) -> SlopeGradientComputationOutput:
        hydro_analysis = HydrologicAnalysis(
            dem, out_dir=self.out_dir, wbw_env=self.wbw_env
        )
        wbw_outlet = Outlets(
            [self.outlet], crs=self.outlet.crs
        ).to_whitebox_vector()
        return hydro_analysis.compute_slope_gradient(
            wbw_outlet,
            streams_flow_accum_threshold=STREAM_FLOW_ACCUM_THRESHOLD,
            outlet_snap_dist=100,
        )

    def get_streams_as_points(
        self, slope_grad: SlopeGradientComputationOutput
    ) -> WhiteboxVector:
        stream_profiles_path = self.out_dir / 'streams.shp'
        stream_profiles = self.wbw_env.raster_to_vector_points(
            slope_grad.streams
        )
        return write_whitebox(
            stream_profiles,
            stream_profiles_path,
            overwrite=True,
            wbw_env=self.wbw_env,
        )

    def rename_profiles_fields(
        self, profiles: Path, raster_names: c.Iterable[str]
    ) -> None:
        # This renames the fields.
        # Can't figure out how to do it with the Whitebox Workflows API
        with redirect_warnings(
            self.logger_adapter, RuntimeWarning, module='pyogrio.raw'
        ):
            gpd.read_file(profiles).rename(
                columns={
                    f'VALUE{i}': raster_name
                    for i, raster_name in enumerate(raster_names, start=1)
                }
            ).to_file(profiles)

    def get_profiles(
        self,
        dem: Path,
        slope_3x3: SAGARaster,
        slope_grad: SlopeGradientComputationOutput,
        streams: WhiteboxVector,
    ) -> Path:
        profiles_path = self.out_dir / 'profiles.shp'
        rasters = {
            'slope': self.wbw_env.read_raster(fspath(slope_3x3.path)),
            'slope_grad': slope_grad.slope_grad,
            'flowacc': slope_grad.flow.flow_accumulation,
        }
        output = self.wbw_env.extract_raster_values_at_points(
            rasters=list(rasters.values()),
            points=streams,
        )[0]
        write_whitebox(
            output, profiles_path, overwrite=True, wbw_env=self.wbw_env
        )

        self.rename_profiles_fields(profiles_path, rasters)
        return profiles_path

    def run(self) -> TrialResult:
        assert self.logger_adapter is not None
        try:
            vrt = self.get_vrt()
            self.logger_adapter.info(
                f'Resampling VRT DEM to resolution {self.resolution}'
            )
            dem = self.get_resampled_dem(vrt)
            self.logger_adapter.info('Computing 3x3 slope')
            slope_grad = self.get_slope_gradient(dem)
            self.logger_adapter.info('Generating stream network')
            slope_3x3 = self.get_3x3_slope(dem)
            self.logger_adapter.info('Computing slope gradient')
            streams = self.get_streams_as_points(slope_grad)
            self.logger_adapter.info('Generating profiles from stream network')
            profiles = self.get_profiles(dem, slope_3x3, slope_grad, streams)
            self.logger_adapter.info('Trial finished with success!')
            ret = TrialResult(
                self.name, gpd.read_file(profiles), self.resolution
            )
        except Exception as e:
            self.logger_adapter.error('Trial failed with error: %s' % e)
            raise e
        return ret


@dataclass
class TrialResult:
    name: str
    profiles: pd.DataFrame
    resolution: Resolution
