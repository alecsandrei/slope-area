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
from os import PathLike, makedirs
from pathlib import Path
import queue
import threading
import time
import traceback
import typing as t

from geopandas import gpd
import pandas as pd
import PySAGA_cmd
from rich.live import Live
from whitebox_workflows.whitebox_workflows import Raster as WhiteboxRaster
from whitebox_workflows.whitebox_workflows import Vector as WhiteboxVector

from slope_area._typing import (
    AnyLogger,
    DEMProvider,
    SlopeProviders,
)
from slope_area.config import (
    IS_NOTEBOOK,
    get_saga_env,
    get_wbw_env,
)
from slope_area.console import (
    RichTableRowData,
    create_rich_logger,
    make_table,
    rich_table_logs_thread,
)
from slope_area.enums import SlopeAreaMethod, TrialStatus
from slope_area.features import Outlet, Outlets, Raster
from slope_area.geomorphometry import (
    DefaultSlopeProviders,
    HydrologicAnalysisConfig,
    StreamsComputationOutput,
    compute_streams,
    mask_dem_with_watershed,
    preprocess_dem,
)
from slope_area.logger import (
    TrialLoggerAdapter,
    create_logger,
    turn_off_handlers,
)
from slope_area.plot import SlopeAreaPlotConfig, slope_area_grid
from slope_area.utils import (
    read_whitebox_raster,
    write_whitebox,
)

if t.TYPE_CHECKING:
    from whitebox_workflows.whitebox_workflows import WbEnvironment

    from slope_area._typing import (
        AnyDEM,
        Resolution,
        RichTableLogs,
        StrPath,
        TrialLoggingContext,
    )


m_logger = create_logger(__name__)


@dataclass(frozen=True)
class BuilderConfig:
    hydrologic_analysis_config: HydrologicAnalysisConfig
    out_dir: StrPath
    out_fig: StrPath
    method: SlopeAreaMethod = SlopeAreaMethod.MAIN_STREAM
    slope_providers: SlopeProviders | None = None
    plot_config: SlopeAreaPlotConfig | None = None
    max_workers: int | None = None

    def __post_init__(self) -> None:
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
    def save_plot(self, trial_results: TrialResults) -> None: ...

    def build(self) -> TrialResults:
        trials = self.get_trials()
        results = trials.run(self.config.max_workers)
        self.save_plot(results)
        return results


@dataclass
class ResolutionPlotBuilder(Builder):
    dem: DEMProvider | Raster | StrPath
    outlet: Outlet
    resolutions: c.Sequence[Resolution]

    @cached_property
    def trial_names(self) -> list[str]:
        return [
            f'Resolution {resolution[0]}' for resolution in self.resolutions
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
                        method=self.config.method,
                        out_dir=Path(self.config.out_dir) / trial_name,
                        slope_providers=self.config.slope_providers,
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
            out_fig=Path(self.config.out_dir) / 'slope_area.png',
            config=plot_config,
        )


@dataclass
class OutletPlotBuilder(Builder):
    dem: AnyDEM
    outlets: Outlets
    resolution: Resolution | None = None

    @cached_property
    def trial_names(self) -> list[str]:
        return [outlet.name for outlet in self.outlets]

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
                        method=self.config.method,
                        out_dir=Path(self.config.out_dir) / trial_name,
                        slope_providers=self.config.slope_providers,
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
            out_fig=Path(self.config.out_dir) / 'slope_area.png',
            config=plot_config,
        )


@dataclass
class TrialConfig:
    name: str
    outlet: Outlet
    dem: DEMProvider | Raster | StrPath
    hydrologic_analysis_config: HydrologicAnalysisConfig
    out_dir: Path
    method: SlopeAreaMethod = SlopeAreaMethod.MAIN_STREAM
    slope_providers: SlopeProviders | None = None
    resolution: Resolution | None = None

    def __post_init__(self) -> None:
        makedirs(self.out_dir, exist_ok=True)


@dataclass(init=False)
class TrialQueue:
    rich: queue.Queue[RichTableLogs | None]
    logging: queue.Queue[logging.LogRecord]

    def __init__(self) -> None:
        self.rich = multiprocessing.Manager().Queue(-1)
        self.logging = multiprocessing.Manager().Queue(-1)


@dataclass
class Trial:
    config: TrialConfig
    logger: logging.Logger = field(init=False, repr=False)
    logger_adapter: TrialLoggerAdapter = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.set_logger()

    @property
    def wbw_env(self) -> WbEnvironment:
        return get_wbw_env()

    @property
    def saga_env(self) -> PySAGA_cmd.SAGA:
        return get_saga_env()

    def set_logger_multiprocess(
        self, q: TrialQueue, default_logs: RichTableLogs
    ) -> None:
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

    def set_logger(self) -> None:
        logger = m_logger.getChild(self.__class__.__name__)
        self.logger = logger
        self.logger_adapter = TrialLoggerAdapter(
            self.logger, trial_name=self.config.name
        )

    def get_dem(self) -> Raster:
        if isinstance(self.config.dem, DEMProvider):
            self.log('Getting the DEM raster from dem_provider')
            return self.config.dem.get_dem(
                outlet=self.config.outlet,
                logger=self.logger_adapter,
            )
        elif isinstance(self.config.dem, (str, PathLike)):
            return Raster(self.config.dem)
        return self.config.dem

    def get_masked_dem_preproc(self, dem: Raster) -> WhiteboxRaster:
        wbw_outlet = Outlets([self.config.outlet]).to_whitebox_vector(dem.crs)
        dem_preproc = preprocess_dem(dem, logger=self.logger_adapter)
        masked_dem = mask_dem_with_watershed(
            dem_preproc,
            outlet=wbw_outlet,
            outlet_snap_distance=self.config.hydrologic_analysis_config.outlet_snap_distance,
            logger=self.logger_adapter,
        )
        write_whitebox(
            masked_dem,
            out_file=self.config.out_dir / 'dem_preproc_mask.tif',
            logger=self.logger_adapter,
            overwrite=True,
        )
        return masked_dem

    def get_resampled_dem(self, raster: Raster) -> Raster:
        assert self.config.resolution is not None
        dem_resampled_path = self.config.out_dir / 'dem_resampled.tif'
        return raster.resample(
            out_file=dem_resampled_path,
            resolution=self.config.resolution,
            logger=self.logger_adapter,
        )

    def get_streams_as_points(self, streams: WhiteboxRaster) -> WhiteboxVector:
        stream_profiles_path = self.config.out_dir / 'streams.shp'
        stream_profiles = self.wbw_env.raster_to_vector_points(streams)
        return write_whitebox(
            stream_profiles,
            stream_profiles_path,
            overwrite=True,
            wbw_env=self.wbw_env,
            logger=self.logger_adapter,
        )

    def get_profiles(
        self,
        slopes: dict[str, StrPath],
        flow_acc: WhiteboxRaster,
        streams: WhiteboxVector,
    ) -> Path:
        self.log('Generating profiles from stream network')
        if 'area' in slopes:
            raise ValueError(
                '"area" is a reserved key and cannot be a slope provider'
            )
        profiles_path = self.config.out_dir / 'profiles.shp'
        rasters = {
            slope_name: read_whitebox_raster(raster, logger=self.logger_adapter)
            for slope_name, raster in slopes.items()
        }
        rasters['area'] = flow_acc
        output = self.wbw_env.extract_raster_values_at_points(
            rasters=list(rasters.values()),
            points=streams,
        )[0]
        write_whitebox(
            output,
            profiles_path,
            overwrite=True,
            logger=self.logger_adapter,
        )

        return profiles_path

    def process_profiles(
        self, profiles: Path, slope_names: c.Sequence[str]
    ) -> gpd.GeoDataFrame:
        self.logger_adapter.info('Reading profiles %s' % profiles)
        gdf = gpd.read_file(profiles)
        raster_names = list(slope_names)
        raster_names.append('area')
        rename_map = {
            f'VALUE{i}': slope_name
            for i, slope_name in enumerate(raster_names, start=1)
        }

        # Rename slopes. Whitebox Tools outputs VALUE1, VALUE2, etc.
        # The last value is the flow accumulation
        gdf = gdf.rename(columns=rename_map)

        gdf = gdf.melt(
            id_vars=gdf.columns.difference(slope_names),
            value_vars=slope_names,
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
    ) -> None:
        assert self.logger is not None
        context: TrialLoggingContext = {}
        if status is not None:
            context.update({'trialStatus': status})
        if exception is not None:
            context.update({'trialException': exception})
        self.logger_adapter.trial_context.update(context)
        self.logger_adapter.log(level=level, msg=msg, stacklevel=2)

    def get_streams(
        self, dem_preproc: WhiteboxRaster
    ) -> StreamsComputationOutput:
        streams = compute_streams(
            dem_preproc,
            self.config.hydrologic_analysis_config.streams_flow_accumulation_threshold,
            main_stream=self.config.method is SlopeAreaMethod.MAIN_STREAM,
            logger=self.logger_adapter,
        )
        write_whitebox(
            streams.streams,
            self.config.out_dir / 'streams.tif',
            logger=self.logger_adapter,
            overwrite=True,
        )
        return streams

    def get_default_slope_providers(
        self, streams_output: StreamsComputationOutput
    ) -> SlopeProviders:
        self.log('Creating default slope providers')
        slope_3x3 = DefaultSlopeProviders.Slope3x3()
        slope_continuous = DefaultSlopeProviders.StreamSlopeContinuous(
            streams_output.flow.d8_pointer,
            streams_output.streams,
            self.config.hydrologic_analysis_config.streams_flow_accumulation_threshold,
        )
        return {
            'Slope3x3': slope_3x3,
            'StreamSlopeContinuous': slope_continuous,
        }

    def compute_slopes(
        self, slope_providers: SlopeProviders, dem_preproc: StrPath
    ) -> dict[str, StrPath]:
        results = {}
        for slope_name, provider in slope_providers.items():
            self.logger_adapter.info(
                'Computing slope %r with provider %r'
                % (slope_name, provider.__class__.__name__)
            )
            slope = provider.get_slope(
                dem_preproc, out_file=self.config.out_dir / f'{slope_name}.tif'
            )
            results[slope_name] = slope
        return results

    def execute(self) -> TrialResult:
        dem = self.get_dem()
        if self.config.resolution is not None:
            dem = self.get_resampled_dem(dem)
        dem_preproc = self.get_masked_dem_preproc(dem)
        streams_output = self.get_streams(dem_preproc)
        streams_vec = self.get_streams_as_points(streams_output.streams)
        slope_providers = self.config.slope_providers
        if slope_providers is None:
            slope_providers = self.get_default_slope_providers(streams_output)
        slopes = self.compute_slopes(
            slope_providers, dem_preproc=dem_preproc.file_name
        )
        profiles = self.get_profiles(
            slopes, streams_output.flow.flow_accumulation, streams_vec
        )
        processed_profiles = self.process_profiles(
            profiles, slope_names=list(slope_providers)
        )
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

    def __post_init__(self, logger: AnyLogger | None) -> None:
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
    ) -> None:
        while any(f.running() for f in futures):
            live.update(make_table(self.logs), refresh=IS_NOTEBOOK)
            time.sleep(0.1)
        live.update(make_table(self.logs), refresh=IS_NOTEBOOK)

    def gather(
        self, futures: c.Sequence[concurrent.futures.Future[TrialResult]]
    ) -> TrialResults:
        self._logger.info('Gathering results of TrialsExecutor')
        results = TrialResults()
        for i, future in enumerate(futures):
            try:
                result = future.result()
            except Exception as e:
                tb_str = ''.join(
                    traceback.format_exception(type(e), e, e.__traceback__)
                )
                self._logger.error(
                    'Trial %r has failed with error\n%s'
                    % (self.trials[i].config.name, tb_str)
                )
            else:
                results.append(result)
        if not results:
            self._logger.error('All %i trials have errored.' % len(self.trials))
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
