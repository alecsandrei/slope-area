from __future__ import annotations

from abc import ABC, abstractmethod
import collections.abc as c
import concurrent.futures
from dataclasses import astuple, dataclass, field, fields
from functools import cached_property
import logging
from logging.handlers import QueueHandler
import multiprocessing
from os import PathLike, fspath, makedirs
from pathlib import Path
import queue
import threading
import time
import typing as t

from geopandas import gpd
import pandas as pd
import PySAGA_cmd
from PySAGA_cmd import Raster as SAGARaster
from rich import box
from rich.live import Live
from rich.table import Table
from whitebox_workflows import WbEnvironment
from whitebox_workflows.whitebox_workflows import Vector as WhiteboxVector

from slope_area._typing import (
    AnyLogger,
    Resolution,
    RichTableLogs,
    TrialLoggingContext,
)
from slope_area.config import (
    IS_NOTEBOOK,
    get_saga_env,
    get_saga_raster_suffix,
    get_wbw_env,
)
from slope_area.enums import TrialStatus
from slope_area.features import (
    VRT,
    DEMSource,
    DEMTiles,
    Outlet,
    Outlets,
    Raster,
)
from slope_area.geomorphometry import (
    HydrologicAnalysis,
    HydrologicAnalysisConfig,
    SlopeGradientComputationOutput,
    compute_slope,
)
from slope_area.logger import (
    RichDictHandler,
    create_logger,
)
from slope_area.plot import (
    SlopeAreaPlotConfig,
    preprocess_trial_results,
    slope_area_grid,
)
from slope_area.utils import (
    redirect_warnings,
    silence_logger_stdout_stderr,
    write_whitebox,
)

m_logger = create_logger(__name__)


@dataclass
class RichTableRowData:
    trial: str
    message: str
    status: str
    exception: str


def make_table(logs: RichTableLogs) -> Table:
    table = Table(box=box.SQUARE_DOUBLE_HEAD, show_header=True, expand=True)

    for dataclass_field in fields(RichTableRowData):
        table.add_column(dataclass_field.name.capitalize())
    for row_data in logs.values():
        table.add_row(*astuple(row_data))
    return table


def create_rich_logger(
    logger_name: str, logs: RichTableLogs, q: queue.Queue
) -> logging.Logger:
    r_logger = m_logger.getChild(logger_name)
    rich_handler = RichDictHandler(logs, q)
    r_logger.addHandler(rich_handler)
    return r_logger


@dataclass(frozen=True)
class BuilderConfig:
    hydrologic_analysis_config: HydrologicAnalysisConfig
    out_dir: Path
    max_workers: int | None = None

    def __post_init__(self):
        makedirs(self.out_dir, exist_ok=True)


def rich_table_logs_thread(logs: RichTableLogs, q: queue.Queue):
    while True:
        trial_log: RichTableLogs = q.get()
        if trial_log is None:
            break
        logs.update(trial_log)


@dataclass
class Builder(ABC):
    config: BuilderConfig

    @property
    @abstractmethod
    def trial_names(self) -> list[str]: ...

    @abstractmethod
    def get_trials(self) -> list[Trial]: ...

    @abstractmethod
    def save_plot(self, trial_results: list[TrialResult]): ...

    def run_trials(
        self, trials: c.Iterable[Trial], logs: RichTableLogs
    ) -> list[concurrent.futures.Future[TrialResult]]:
        with (
            Live(
                make_table(logs),
                refresh_per_second=10,
                redirect_stderr=True,
                redirect_stdout=True,
            ) as live,
            silence_logger_stdout_stderr('slopeArea'),
        ):
            q = TrialQueue()
            make_table_thread = threading.Thread(
                target=rich_table_logs_thread,
                args=(logs, q.rich),
            )
            make_table_thread.start()
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.config.max_workers
            ) as executor:
                futures = [
                    executor.submit(trial.run, q, logs) for trial in trials
                ]
                while any(f.running() for f in futures):
                    live.update(make_table(logs), refresh=IS_NOTEBOOK)
                    time.sleep(0.1)
                live.update(make_table(logs), refresh=IS_NOTEBOOK)
            q.rich.put(None)
            make_table_thread.join()
        return futures

    @staticmethod
    def get_default_log_for_trial(
        trial_name: str,
    ) -> RichTableRowData:
        return RichTableRowData(
            trial=trial_name,
            message='[dim]Waiting...[/dim]',
            status=TrialStatus.NOT_STARTED.display(),
            exception='',
        )

    def build(self) -> None:
        logs: RichTableLogs = {
            name: self.get_default_log_for_trial(name)
            for name in self.trial_names
        }
        # m_logger = create_rich_logger(self.__class__.__name__, logs)
        trials = self.get_trials()
        futures = self.run_trials(trials, logs)
        trial_results = [
            future.result() for future in futures if future.exception() is None
        ]
        self.save_plot(trial_results)


@dataclass
class ResolutionPlotBuilder(Builder):
    dem_source: DEMSource
    outlet: Outlet
    resolutions: c.Sequence[Resolution]

    def get_vrt(self) -> VRT:
        dem = self.config.out_dir / 'dem.vrt'
        dem_tiles = DEMTiles.from_outlet(
            dem_source=self.dem_source,
            outlet=self.outlet,
            out_dir=self.config.out_dir,
            outlet_snap_dist=self.config.hydrologic_analysis_config.outlet_snap_distance,
        )
        return VRT.from_dem_tiles(dem_tiles, dem)

    @cached_property
    def trial_names(self) -> list[str]:
        unit_name = self.outlet.crs.axis_info[0].unit_name
        return [
            f'{resolution[0]} {unit_name}' for resolution in self.resolutions
        ]

    def get_trials(self) -> list[Trial]:
        vrt = self.get_vrt()
        return [
            Trial(
                config=TrialConfig(
                    name=trial_name,
                    outlet=self.outlet,
                    resolution=resolution,
                    hydrologic_analysis_config=self.config.hydrologic_analysis_config,
                    dem_provider=StaticVRT(vrt),
                    out_dir=self.config.out_dir / trial_name,
                ),
                # logger=logger,
            )
            for trial_name, resolution in zip(
                self.trial_names, self.resolutions
            )
        ]

    def save_plot(self, trial_results: c.Iterable[TrialResult]) -> None:
        slope_area_grid(
            data=preprocess_trial_results(trial_results),
            col='trial_name',
            out_fig=self.config.out_dir / 'slope_area.png',
            config=SlopeAreaPlotConfig(),
        )


def get_default_log_for_trial(trial_name: str) -> RichTableRowData:
    return RichTableRowData(
        trial=trial_name,
        message='[dim]Waiting...[/dim]',
        status=TrialStatus.NOT_STARTED.display(),
        exception='',
    )


@dataclass
class OutletPlotBuilder(Builder):
    dem_source: DEMSource
    outlets: Outlets
    resolution: Resolution

    @cached_property
    def trial_names(self) -> list[str]:
        return [str(outlet) for outlet in self.outlets]

    def get_vrt(self, wbw_env: WbEnvironment, out_dir: Path, outlet: Outlet):
        dem_tiles = DEMTiles.from_outlet(
            dem_source=self.dem_source,
            outlet=outlet,
            out_dir=out_dir,
            outlet_snap_dist=self.config.hydrologic_analysis_config.outlet_snap_distance,
            wbw_env=wbw_env,
        )

        return VRT.from_dem_tiles(dem_tiles, self.config.out_dir / 'dem.vrt')

    def get_trials(self) -> list[Trial]:
        return [
            Trial(
                config=TrialConfig(
                    name=trial_name,
                    outlet=outlet,
                    resolution=self.resolution,
                    dem_provider=DynamicVRT(
                        self.dem_source,
                        outlet,
                        self.config.out_dir / trial_name,
                        outlet_snap_distance=self.config.hydrologic_analysis_config.outlet_snap_distance,
                    ),
                    hydrologic_analysis_config=self.config.hydrologic_analysis_config,
                    out_dir=self.config.out_dir / trial_name,
                ),
                # logger=logger,
            )
            for trial_name, outlet in zip(self.trial_names, self.outlets)
        ]

    def save_plot(self, trial_results: c.Iterable[TrialResult]) -> None:
        slope_area_grid(
            data=preprocess_trial_results(trial_results),
            col='trial_name',
            out_fig=self.config.out_dir / 'slope_area.png',
            config=SlopeAreaPlotConfig(),
        )


class DEMProvider(t.Protocol):
    def get_dem(
        self,
        *,
        wbw_env: WbEnvironment,
        logger: AnyLogger | None = None,
    ) -> Raster: ...


@dataclass
class StaticVRT(DEMProvider):
    vrt: VRT

    def get_dem(self, *args, **kwargs) -> VRT:
        return self.vrt


@dataclass
class DynamicVRT(DEMProvider):
    dem_source: DEMSource
    outlet: Outlet
    out_dir: PathLike
    outlet_snap_distance: float

    def get_dem(
        self,
        *,
        wbw_env: WbEnvironment,
        logger: AnyLogger | None = None,
    ):
        if wbw_env is None:
            raise
        dem_tiles = DEMTiles.from_outlet(
            dem_source=self.dem_source,
            outlet=self.outlet,
            out_dir=self.out_dir,
            outlet_snap_dist=self.outlet_snap_distance,
            wbw_env=wbw_env,
            logger=logger,
        )

        return VRT.from_dem_tiles(dem_tiles, Path(self.out_dir) / 'dem.vrt')


@dataclass
class TrialConfig:
    name: str
    outlet: Outlet
    resolution: Resolution
    hydrologic_analysis_config: HydrologicAnalysisConfig
    dem_provider: DEMProvider
    out_dir: Path

    def __post_init__(self):
        makedirs(self.out_dir, exist_ok=True)


@dataclass(init=False)
class TrialQueue:
    rich: queue.Queue
    logging: queue.Queue

    def __init__(self):
        self.rich = multiprocessing.Manager().Queue(-1)
        self.logging = multiprocessing.Manager().Queue(-1)


class TrialLoggerAdapter(logging.LoggerAdapter):
    def __init__(
        self,
        logger,
        trial_name: str,
        trial_context: TrialLoggingContext | None = None,
    ):
        super().__init__(logger)
        self.logger = logger
        self.trial_name = trial_name
        self.trial_context = trial_context or {}

    def process(
        self, msg: t.Any, kwargs: c.MutableMapping[str, t.Any]
    ) -> tuple[t.Any, c.MutableMapping[str, t.Any]]:
        kwargs.setdefault('extra', {}).update(
            {'trialName': self.trial_name, 'trialContext': self.trial_context}
        )
        return (msg, kwargs)


@dataclass
class Trial:
    config: TrialConfig
    logger: logging.Logger | None = field(init=False, repr=False)
    logger_adapter: TrialLoggerAdapter = field(init=False, repr=False)
    _wbw_env: WbEnvironment | None = field(init=False, default=None, repr=False)
    _saga_env: PySAGA_cmd.SAGA | None = field(
        init=False, default=None, repr=False
    )

    @property
    def wbw_env(self) -> WbEnvironment:
        if self._wbw_env is None:
            self._wbw_env = get_wbw_env(self.logger_adapter)
        return self._wbw_env

    @property
    def saga_env(self) -> PySAGA_cmd.SAGA:
        if self._saga_env is None:
            self._saga_env = get_saga_env(self.logger_adapter)
        return self._saga_env

    def get_resampled_dem(self, raster: Raster) -> Path:
        dem_resampled_path = self.config.out_dir / 'dem_resampled.tif'
        crs = raster.crs
        if not crs.is_projected:
            self.log(
                'CRS of %s is unprojected. Defaulting to the outlet CRS EPSG:%s'
                % (Path(raster.path).name, self.config.outlet.crs.to_epsg()),
                level=logging.WARNING,
            )
            crs = self.config.outlet.crs
        return raster.resample(
            out_file=dem_resampled_path,
            resolution=self.config.resolution,
            crs=crs,
            logger=self.logger_adapter,
        )

    def get_3x3_slope(self, dem: Path) -> SAGARaster:
        slope_3x3_path = (self.config.out_dir / 'slope').with_suffix(
            get_saga_raster_suffix(self.saga_env)
        )
        return compute_slope(
            dem,
            slope_3x3_path,
            saga_env=self.saga_env,
            logger=self.logger_adapter,
        )

    def get_slope_gradient(self, dem: Path) -> SlopeGradientComputationOutput:
        hydro_analysis = HydrologicAnalysis(
            dem,
            out_dir=self.config.out_dir,
            wbw_env=self.wbw_env,
            logger=self.logger_adapter,
        )
        wbw_outlet = Outlets(
            [self.config.outlet], crs=self.config.outlet.crs
        ).to_whitebox_vector()
        return hydro_analysis.compute_slope_gradient(
            wbw_outlet, config=self.config.hydrologic_analysis_config
        )

    def get_streams_as_points(
        self, slope_grad: SlopeGradientComputationOutput
    ) -> WhiteboxVector:
        stream_profiles_path = self.config.out_dir / 'streams.shp'
        stream_profiles = self.wbw_env.raster_to_vector_points(
            slope_grad.streams
        )
        return write_whitebox(
            stream_profiles,
            stream_profiles_path,
            overwrite=True,
            wbw_env=self.wbw_env,
            logger=self.logger_adapter,
        )

    def rename_profiles_fields(
        self, profiles: Path, raster_names: c.Iterable[str]
    ) -> None:
        # This renames the fields.
        # Can't figure out how to do it with the Whitebox Workflows API
        assert self.logger is not None
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
        slope_3x3: SAGARaster,
        slope_grad: SlopeGradientComputationOutput,
        streams: WhiteboxVector,
    ) -> Path:
        profiles_path = self.config.out_dir / 'profiles.shp'
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
            output,
            profiles_path,
            overwrite=True,
            wbw_env=self.wbw_env,
            logger=self.logger_adapter,
        )

        self.rename_profiles_fields(profiles_path, rasters)
        return profiles_path

    def log(
        self,
        msg: str = '',
        level: int = logging.INFO,
        status: TrialStatus | None = None,
        exception: Exception | None = None,
    ):
        assert self.logger is not None
        context: TrialLoggingContext = {}
        if status is not None:
            context.update({'trialStatus': status})
        if exception is not None:
            context.update({'trialException': exception})
        self.logger_adapter.trial_context.update(context)
        self.logger_adapter.log(level=level, msg=msg, stacklevel=2)

    def set_logger(self, q: TrialQueue, default_logs: RichTableLogs):
        qh = QueueHandler(q.logging)
        logger = create_rich_logger(
            self.__class__.__name__, default_logs, q.rich
        )
        logger.addHandler(qh)
        logger.setLevel(logging.DEBUG)
        self.logger = logger
        self.logger_adapter = TrialLoggerAdapter(
            self.logger, trial_name=self.config.name
        )

    def run(self, q: TrialQueue, default_logs: RichTableLogs) -> TrialResult:
        self.set_logger(q, default_logs)
        try:
            self.log(status=TrialStatus.RUNNING)
            self.log('Getting the DEM raster')
            dem = self.config.dem_provider.get_dem(
                wbw_env=self.wbw_env, logger=self.logger_adapter
            )
            self.log(
                'Resampling DEM %s to resolution %s'
                % (Path(dem.path).name, self.config.resolution)
            )
            dem_resampled = self.get_resampled_dem(dem)
            self.log('Computing 3x3 slope')
            slope_grad = self.get_slope_gradient(dem_resampled)
            self.log('Generating stream network')
            slope_3x3 = self.get_3x3_slope(dem_resampled)
            self.log('Computing slope gradient')
            streams = self.get_streams_as_points(slope_grad)
            self.log('Generating profiles from stream network')
            profiles = self.get_profiles(slope_3x3, slope_grad, streams)
            self.log('Reading the profiles %s' % profiles.name)
            ret = TrialResult(gpd.read_file(profiles), self.config)
        except Exception as e:
            self.log(
                'Trial failed with error: %s' % e,
                level=logging.ERROR,
                status=TrialStatus.ERRORED,
                exception=e,
            )
            raise e
        else:
            self.log(
                'Trial finished with success!', status=TrialStatus.FINISHED
            )
            return ret


@dataclass
class TrialResult:
    profiles: pd.DataFrame
    config: TrialConfig
