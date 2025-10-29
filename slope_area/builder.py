from __future__ import annotations

from abc import ABC, abstractmethod
from collections import UserList
import collections.abc as c
import concurrent.futures
from dataclasses import InitVar, dataclass, field
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
from rich.live import Live
from whitebox_workflows import WbEnvironment
from whitebox_workflows.whitebox_workflows import Vector as WhiteboxVector

from slope_area._typing import AnyLogger, DEMProvider
from slope_area.config import (
    IS_NOTEBOOK,
    get_saga_env,
    get_saga_raster_suffix,
    get_wbw_env,
)
from slope_area.console import (
    RichTableRowData,
    create_rich_logger,
    make_table,
    rich_table_logs_thread,
)
from slope_area.enums import TrialStatus
from slope_area.features import Outlet, Outlets, Raster
from slope_area.geomorphometry import (
    HydrologicAnalysis,
    HydrologicAnalysisConfig,
    SlopeGradientComputationOutput,
    compute_slope,
)
from slope_area.logger import (
    TrialLoggerAdapter,
    create_logger,
    turn_off_handlers,
)
from slope_area.plot import SlopeAreaPlotConfig, slope_area_grid
from slope_area.utils import redirect_warnings, write_whitebox

if t.TYPE_CHECKING:
    from slope_area._typing import (
        AnyDEM,
        Resolution,
        RichTableLogs,
        TrialLoggingContext,
    )


m_logger = create_logger(__name__)


@dataclass(frozen=True)
class BuilderConfig:
    hydrologic_analysis_config: HydrologicAnalysisConfig
    out_dir: PathLike
    out_fig: PathLike
    plot_config: SlopeAreaPlotConfig | None = None
    max_workers: int | None = None

    def __post_init__(self):
        makedirs(self.out_dir, exist_ok=True)


@dataclass
class Builder(ABC):
    config: BuilderConfig

    @property
    @abstractmethod
    def trial_names(self) -> list[str]: ...

    @abstractmethod
    def get_trials(self) -> Trials: ...

    @abstractmethod
    def save_plot(self, trial_results: TrialResults): ...

    def build(self) -> TrialResults:
        trials = self.get_trials()
        results = trials.run(self.config.max_workers)
        self.save_plot(results)
        return results


@dataclass
class ResolutionPlotBuilder(Builder):
    dem: DEMProvider | Raster | PathLike
    outlet: Outlet
    resolutions: c.Sequence[Resolution]

    @cached_property
    def trial_names(self) -> list[str]:
        unit_name = self.outlet.crs.axis_info[0].unit_name
        return [
            f'{resolution[0]} {unit_name}' for resolution in self.resolutions
        ]

    def get_trials(self) -> Trials:
        return Trials(
            [
                Trial(
                    config=TrialConfig(
                        name=trial_name,
                        outlet=self.outlet,
                        resolution=resolution,
                        hydrologic_analysis_config=self.config.hydrologic_analysis_config,
                        dem=self.dem,
                        out_dir=self.config.out_dir / trial_name,
                    ),
                )
                for trial_name, resolution in zip(
                    self.trial_names, self.resolutions
                )
            ]
        )

    def save_plot(self, trial_results: TrialResults) -> None:
        plot_config = self.config.plot_config
        if plot_config is None:
            plot_config = SlopeAreaPlotConfig(
                hue='slope_type',
                col='trial',
            )
        slope_area_grid(
            data=trial_results.to_dataframe(),
            out_fig=self.config.out_dir / 'slope_area.png',
            config=plot_config,
        )


@dataclass
class OutletPlotBuilder(Builder):
    dem: AnyDEM
    outlets: Outlets
    resolution: Resolution | None = None

    @cached_property
    def trial_names(self) -> list[str]:
        return [str(outlet) for outlet in self.outlets]

    def get_trials(self) -> Trials:
        return Trials(
            [
                Trial(
                    config=TrialConfig(
                        name=trial_name,
                        outlet=outlet,
                        resolution=self.resolution,
                        dem=self.dem,
                        hydrologic_analysis_config=self.config.hydrologic_analysis_config,
                        out_dir=self.config.out_dir / trial_name,
                    ),
                )
                for trial_name, outlet in zip(self.trial_names, self.outlets)
            ]
        )

    def save_plot(self, trial_results: TrialResults) -> None:
        plot_config = self.config.plot_config
        if plot_config is None:
            plot_config = SlopeAreaPlotConfig(
                hue='slope_type',
                col='trial',
            )
        slope_area_grid(
            data=trial_results.to_dataframe(),
            out_fig=self.config.out_dir / 'slope_area.png',
            config=plot_config,
        )


@dataclass
class TrialConfig:
    name: str
    outlet: Outlet
    dem: DEMProvider | Raster | PathLike
    hydrologic_analysis_config: HydrologicAnalysisConfig
    out_dir: Path
    resolution: Resolution | None = None

    def __post_init__(self):
        makedirs(self.out_dir, exist_ok=True)


@dataclass(init=False)
class TrialQueue:
    rich: queue.Queue
    logging: queue.Queue

    def __init__(self):
        self.rich = multiprocessing.Manager().Queue(-1)
        self.logging = multiprocessing.Manager().Queue(-1)


@dataclass
class Trial:
    config: TrialConfig
    logger: logging.Logger | None = field(default=None, repr=False)
    logger_adapter: TrialLoggerAdapter = field(init=False, repr=False)
    _wbw_env: WbEnvironment | None = field(init=False, default=None, repr=False)
    _saga_env: PySAGA_cmd.SAGA | None = field(
        init=False, default=None, repr=False
    )

    # __gestate__ and __setstate__ make this object picklable

    def __getstate__(self) -> dict[str, t.Any]:
        # Return a shallow copy of __dict__ minus unpickleable fields
        state = self.__dict__.copy()
        # Remove unpicklable or environment-specific attributes
        state.pop('_wbw_env', None)
        state.pop('_saga_env', None)
        return state

    def __setstate__(self, state):
        # Restore the pickled state
        self.__dict__.update(state)
        # Reinitialize transient attributes
        self._wbw_env = None
        self._saga_env = None

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

    def get_dem(self) -> Raster:
        if isinstance(self.config.dem, DEMProvider):
            self.log('Getting the DEM raster from dem_provider')
            return self.config.dem.get_dem(
                outlet=self.config.outlet,
                wbw_env=self.wbw_env,
                logger=self.logger_adapter,
            )
        elif isinstance(self.config.dem, PathLike):
            return Raster(self.config.dem)
        return self.config.dem

    def get_resampled_dem(self, raster: Raster) -> Raster:
        assert self.config.resolution is not None
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

    def get_3x3_slope(self, dem: PathLike) -> SAGARaster:
        slope_3x3_path = (self.config.out_dir / 'slope').with_suffix(
            get_saga_raster_suffix(self.saga_env)
        )
        return compute_slope(
            dem,
            slope_3x3_path,
            saga_env=self.saga_env,
            logger=self.logger_adapter,
        )

    def get_slope_gradient(
        self, dem: PathLike
    ) -> SlopeGradientComputationOutput:
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

    def get_main_stream_as_points(
        self, slope_grad: SlopeGradientComputationOutput
    ) -> WhiteboxVector:
        main_stream_path = self.config.out_dir / 'main_stream.shp'
        main_stream_profiles = self.wbw_env.raster_to_vector_points(
            slope_grad.main_stream
        )
        return write_whitebox(
            main_stream_profiles,
            main_stream_path,
            overwrite=True,
            wbw_env=self.wbw_env,
            logger=self.logger_adapter,
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
        self.log('Generating profiles from stream network')
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

    def process_profiles(self, profiles: Path) -> gpd.GeoDataFrame:
        self.logger_adapter.info('Reading profiles %s' % profiles)
        gdf = gpd.read_file(profiles)
        slope_cols = ['Slope 3x3', 'StreamSlopeContinuous']
        gdf = gdf.rename(
            columns={
                'slope_grad': 'StreamSlopeContinuous',
                'slope': 'Slope 3x3',
                'flowacc': 'area',
            }
        )
        gdf = gdf.melt(
            id_vars=gdf.columns.difference(slope_cols),
            value_vars=slope_cols,
            var_name='slope_type',
            value_name='values',
        )
        slope_inv = gdf['values'] / 100
        gdf['values'] = slope_inv
        gdf = gdf.rename(columns={'values': 'slope'})
        gdf['resolution'] = str(self.config.resolution)
        return gdf

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

    def set_logger_multiprocess(
        self, q: TrialQueue, default_logs: RichTableLogs
    ):
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

    def set_logger(self):
        if self.logger is None:
            logger = m_logger.getChild(self.__class__.__name__)
            self.logger = logger
        self.logger_adapter = TrialLoggerAdapter(
            self.logger, trial_name=self.config.name
        )

    def execute(self) -> TrialResult:
        dem = self.get_dem()
        if self.config.resolution is not None:
            dem = self.get_resampled_dem(dem)
        slope_grad = self.get_slope_gradient(dem.path)
        slope_3x3 = self.get_3x3_slope(dem.path)
        # streams = self.get_streams_as_points(slope_grad)
        streams = self.get_main_stream_as_points(slope_grad)
        profiles = self.get_profiles(slope_3x3, slope_grad, streams)
        processed_profiles = self.process_profiles(profiles)
        return TrialResult(processed_profiles, self.config)

    def run_multiprocess(
        self, q: TrialQueue, default_logs: RichTableLogs
    ) -> TrialResult:
        self.set_logger_multiprocess(q, default_logs)
        with turn_off_handlers('slopeArea', ('stdout', 'stderr')):
            return self.run()

    def run(self) -> TrialResult:
        if self.logger is None:
            self.set_logger()
        try:
            self.logger_adapter.mark_running()
            ret = self.execute()
        except Exception as e:
            self.logger_adapter.mark_error(exc=e)
            raise e
        else:
            self.logger_adapter.mark_finished()
        finally:
            return ret


class Trials(UserList[Trial]):
    def run(self, max_workers: int | None = None) -> TrialResults:
        executor = TrialsExecutor(self, max_workers)
        return executor.run()


@dataclass
class TrialsExecutor:
    trials: Trials
    max_workers: int | None = None
    _logger: AnyLogger = field(repr=False, init=False)
    logs: RichTableLogs = field(init=False, repr=False)
    q: TrialQueue = field(init=False, repr=False)
    logger: InitVar[AnyLogger | None] = field(kw_only=True, default=None)

    def __post_init__(self, logger: AnyLogger | None):
        self._logger = logger or m_logger.getChild(self.__class__.__name__)
        self.logs = self.get_rich_init_logs()
        self.q = TrialQueue()

    def get_rich_init_logs(self) -> RichTableLogs:
        return {
            trial.config.name: RichTableRowData.from_trial_name(
                trial.config.name
            )
            for trial in self.trials
        }

    def submit_trials(
        self, executor: concurrent.futures.Executor
    ) -> list[concurrent.futures.Future[TrialResult]]:
        return [
            executor.submit(trial.run_multiprocess, self.q, self.logs)
            for trial in self.trials
        ]

    def update_loop(
        self,
        live: Live,
        futures: c.Sequence[concurrent.futures.Future[TrialResult]],
    ):
        while any(f.running() for f in futures):
            live.update(make_table(self.logs), refresh=IS_NOTEBOOK)
            time.sleep(0.1)
        live.update(make_table(self.logs), refresh=IS_NOTEBOOK)

    def gather(
        self, futures: c.Sequence[concurrent.futures.Future[TrialResult]]
    ) -> TrialResults:
        self._logger.info('Gathering results of trial executor')
        results = TrialResults()
        for i, future in enumerate(futures):
            if (exc := future.exception()) is not None:
                self._logger.debug(
                    'Trial %s failed with error %s' % (self.trials[i], exc)
                )
            results.append(future.result())
        return results

    def run(self) -> TrialResults:
        make_table_thread = threading.Thread(
            target=rich_table_logs_thread,
            args=(self.logs, self.q.rich),
        )
        with Live(
            make_table(self.logs),
            refresh_per_second=10,
            redirect_stderr=True,
            redirect_stdout=True,
        ) as live:
            make_table_thread.start()
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                futures = self.submit_trials(executor)
                self.update_loop(live, futures)

            # stop table thread
            self.q.rich.put(None)
            make_table_thread.join()
        return self.gather(futures)


@dataclass
class TrialResult:
    profiles: pd.DataFrame
    config: TrialConfig


class TrialResults(UserList[TrialResult]):
    def to_dataframe(self) -> pd.DataFrame:
        return pd.concat(
            [r.profiles.assign(trial=r.config.name) for r in self.data]
        )
