from __future__ import annotations

from abc import ABC, abstractmethod
from collections import UserList
import collections.abc as c
import concurrent.futures
from dataclasses import dataclass, field
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

import numpy as np
import pandas as pd
import PySAGA_cmd
import rasterio as rio
from rich.live import Live
from whitebox_workflows.whitebox_workflows import Raster as WhiteboxRaster

from slope_area._typing import DEMProvider, SlopeProvider, StreamSlopeProvider
from slope_area.config import IS_NOTEBOOK, get_saga_env, get_wbw_env
from slope_area.console import (
    RichTableRowData,
    create_rich_logger,
    make_table,
    rich_table_logs_thread,
)
from slope_area.enums import Column, SlopeAreaMethod, TrialStatus
from slope_area.features import Outlet, Outlets, Raster
from slope_area.geomorphometry import (
    DefaultSlopeProviders,
    StreamsComputationOutput,
    compute_flow,
    compute_streams,
    compute_watershed,
    preprocess_dem,
)
from slope_area.logger import (
    TrialLoggerAdapter,
    create_logger,
    turn_off_handlers,
)
from slope_area.plot import (
    SlopeAreaPlotConfig,
    slope_area_grid,
    slope_area_plot_single,
)
from slope_area.utils import mask_raster, write_whitebox

if t.TYPE_CHECKING:
    from matplotlib.axes import Axes
    import seaborn as sns
    from whitebox_workflows.whitebox_workflows import WbEnvironment

    from slope_area._typing import (
        AnyDEM,
        Resolution,
        RichTableLogs,
        SlopeProviders,
        StrPath,
        TrialLoggingContext,
    )


m_logger = create_logger(__name__)


@dataclass
class HydrologicAnalysisConfig:
    streams_flow_accumulation_threshold: int = 1000
    outlet_snap_distance: int = 100


@dataclass
class AnalysisConfig:
    method: SlopeAreaMethod = SlopeAreaMethod.STREAMS
    hydrologic: HydrologicAnalysisConfig = field(
        default_factory=HydrologicAnalysisConfig
    )
    slope_providers: SlopeProviders = field(
        default_factory=DefaultSlopeProviders.get_default_providers
    )


@dataclass
class TrialFactoryContext:
    dem: AnyDEM
    out_dir: Path
    analysis: AnalysisConfig


@dataclass
class TrialFactory(ABC):
    context: TrialFactoryContext

    @abstractmethod
    def generate(self) -> Trials: ...


@dataclass
class ResolutionTrialFactory(TrialFactory):
    outlet: Outlet
    resolutions: c.Sequence[Resolution]

    def generate(self) -> Trials:
        trial_names = [
            f'Resolution {resolution[0]}' for resolution in self.resolutions
        ]
        return Trials(
            [
                Trial(
                    name=trial_name,
                    context=TrialContext(
                        Path(self.context.out_dir) / trial_name,
                        data=TrialData(
                            outlet=self.outlet,
                            dem=self.context.dem,
                            resolution=resolution,
                        ),
                        analysis=self.context.analysis,
                    ),
                )
                for resolution, trial_name in zip(self.resolutions, trial_names)
            ]
        )


@dataclass
class OutletTrialFactory(TrialFactory):
    outlets: Outlets
    resolution: Resolution | None = None

    def generate(self) -> Trials:
        trial_names = [f'Outlet {outlet.name}' for outlet in self.outlets]
        return Trials(
            [
                Trial(
                    name=trial_name,
                    context=TrialContext(
                        Path(self.context.out_dir) / trial_name,
                        data=TrialData(
                            outlet=outlet,
                            dem=self.context.dem,
                            resolution=self.resolution,
                        ),
                        analysis=self.context.analysis,
                    ),
                )
                for outlet, trial_name in zip(self.outlets, trial_names)
            ]
        )


@dataclass(init=False)
class TrialQueue:
    # Used to send logs for a Rich Table from another process
    rich: queue.Queue[RichTableLogs | None]

    # Used to send LogRecords to the main process for
    # a logging File Handler
    logging: queue.Queue[logging.LogRecord]

    def __init__(self) -> None:
        self.rich = multiprocessing.Manager().Queue(-1)
        self.logging = multiprocessing.Manager().Queue(-1)


@dataclass
class TrialData:
    outlet: Outlet
    dem: AnyDEM

    # 'None' will use dem resolution without resampling
    resolution: Resolution | None = None


@dataclass
class TrialContext:
    out_dir: StrPath
    data: TrialData
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    def __post_init__(self) -> None:
        makedirs(self.out_dir, exist_ok=True)


@dataclass
class Trial:
    name: str
    context: TrialContext
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

    @property
    def out_dir(self) -> Path:
        return Path(self.context.out_dir)

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
            self.logger, trial_name=self.name
        )

    def set_logger(self) -> None:
        logger = m_logger.getChild(self.__class__.__name__)
        self.logger = logger
        self.logger_adapter = TrialLoggerAdapter(
            self.logger, trial_name=self.name
        )

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

    def get_dem(self) -> Raster:
        dem = self.context.data.dem
        if isinstance(dem, DEMProvider):
            self.log('Getting the DEM raster from dem_provider')
            return dem.get_dem(
                outlet=self.context.data.outlet,
                logger=self.logger_adapter,
            )
        elif isinstance(dem, (str, PathLike)):
            return Raster(dem)
        return dem

    def get_dem_preproc_masked(self, dem: Raster) -> WhiteboxRaster:
        dem_preproc = preprocess_dem(dem, logger=self.logger_adapter)
        flow_acc = compute_flow(dem_preproc, logger=self.logger_adapter)

        wbw_outlet = Outlets([self.context.data.outlet]).to_whitebox_vector(
            dem.crs
        )
        watershed = compute_watershed(
            flow_acc,
            wbw_outlet,
            outlet_snap_distance=self.context.analysis.hydrologic.outlet_snap_distance,
            logger=self.logger_adapter,
        )
        return mask_raster(
            dem_preproc, watershed.watershed, logger=self.logger_adapter
        )

    def prepare_dem(self) -> WhiteboxRaster:
        dem = self.get_dem()
        if self.context.data.resolution is not None:
            dem = self.get_resampled_dem(dem)
        dem_masked = self.get_dem_preproc_masked(dem)
        write_whitebox(
            dem_masked,
            out_file=self.out_dir / 'dem_preproc_mask.tif',
            logger=self.logger_adapter,
        )
        return dem_masked

    def get_resampled_dem(self, raster: Raster) -> Raster:
        assert self.context.data.resolution is not None
        dem_resampled_path = self.out_dir / 'dem_resampled.tif'
        return raster.resample(
            out_file=dem_resampled_path,
            resolution=self.context.data.resolution,
            logger=self.logger_adapter,
        )

    def get_streams(
        self, dem_preproc: WhiteboxRaster
    ) -> StreamsComputationOutput:
        flow = compute_flow(dem_preproc, logger=self.logger_adapter)
        main_stream = (
            self.context.analysis.method is SlopeAreaMethod.MAIN_STREAM
        )
        streams = compute_streams(
            flow,
            self.context.analysis.hydrologic.streams_flow_accumulation_threshold,
            main_stream=main_stream,
            logger=self.logger_adapter,
        )
        write_whitebox(
            streams.streams,
            self.out_dir / 'streams.tif',
            logger=self.logger_adapter,
        )
        write_whitebox(
            streams.flow.flow_accumulation,
            self.out_dir / 'flow_accumulation.tif',
            logger=self.logger_adapter,
        )
        return streams

    def compute_slopes(
        self,
        dem_preproc: StrPath,
        streams_computation_output: StreamsComputationOutput,
    ) -> dict[str, StrPath]:
        results = {}
        for (
            slope_name,
            provider,
        ) in self.context.analysis.slope_providers.items():
            self.log(
                'Computing slope %r with provider %r'
                % (slope_name, provider.__class__.__name__)
            )
            out_file = self.out_dir / f'{slope_name}.tif'
            if isinstance(provider, SlopeProvider):
                slope = provider.get_slope(dem_preproc, out_file=out_file)
            elif isinstance(provider, StreamSlopeProvider):
                slope = provider.get_stream_slope(
                    streams_computation_output, out_file=out_file
                )
            else:
                raise ValueError(f'Did not expect provider {provider}')
            results[slope_name] = slope
        return results

    def get_raster_values(
        self,
        rasters: dict[str, StrPath],
        mask: StrPath | None = None,
    ) -> pd.DataFrame:
        raster_file_names = ', '.join(
            Path(raster).name for raster in rasters.values()
        )
        self.log('Extracting raster values from %s' % raster_file_names)

        df = pd.DataFrame(columns=list(rasters))
        if mask is not None:
            with rio.open(mask) as mask_src:
                arr_mask: np.ma.MaskedArray = mask_src.read(1, masked=True)

        for name, raster in rasters.items():
            with rio.open(raster) as src:
                arr: np.ma.MaskedArray = src.read(1, masked=True)
                assert arr.shape == arr_mask.shape, (
                    f'{raster} does not have the same shape as {mask}: {arr.shape}, {arr_mask.shape}'
                )
                if mask is not None:
                    arr.mask = arr.mask | arr_mask.mask
                arr_valid = arr[~arr.mask]
                df[name] = arr_valid

        df.to_csv(self.out_dir / 'values.csv', index=False)
        return df

    def process_raster_values(
        self, values: pd.DataFrame, names: c.Sequence[str]
    ) -> pd.DataFrame:
        self.log('Processing raster values')
        values = values.melt(
            id_vars=values.columns.difference(names),
            value_vars=names,
            var_name=Column.SLOPE_TYPE,
            value_name=Column.SLOPE_VALUES,
        )
        values[Column.SLOPE_VALUES] = values[Column.SLOPE_VALUES] / 100
        values[Column.RESOLUTION] = str(self.context.data.resolution)
        values[Column.TRIAL_NAME] = self.name
        values.to_csv(self.out_dir / 'values_processed.csv', index=False)
        return values

    def execute(self) -> TrialResult:
        dem = self.prepare_dem()
        streams_output = self.get_streams(dem)
        slopes = self.compute_slopes(
            dem_preproc=dem.file_name,
            streams_computation_output=streams_output,
        )
        raster_values = self.get_raster_values(
            slopes | {'area': streams_output.flow.flow_accumulation.file_name},
            streams_output.streams.file_name,
        )
        processed = self.process_raster_values(
            raster_values, names=list(self.context.analysis.slope_providers)
        )
        return TrialResult(processed, self.context)

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
    logs: RichTableLogs = field(init=False, repr=False)
    q: TrialQueue = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.logger = m_logger.getChild(self.__class__.__name__)
        self.logs = self.get_rich_init_logs()
        self.q = TrialQueue()

    def get_rich_init_logs(self) -> RichTableLogs:
        return {
            trial.name: RichTableRowData.from_trial_name(trial.name)
            for trial in self.trials
        }

    def run_trial_in_process(self, trial: Trial) -> TrialResult:
        trial.set_logger_multiprocess(self.q, self.logs)
        with turn_off_handlers('slopeArea', ('stdout', 'stderr')):
            return trial.run()

    def submit_trials(
        self, executor: concurrent.futures.Executor
    ) -> list[concurrent.futures.Future[TrialResult]]:
        return [
            executor.submit(self.run_trial_in_process, trial)
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
        self.logger.info('Gathering results of TrialsExecutor')
        results = TrialResults()
        for i, future in enumerate(futures):
            try:
                result = future.result()
            except Exception as e:
                tb_str = ''.join(
                    traceback.format_exception(type(e), e, e.__traceback__)
                )
                self.logger.error(
                    'Trial %r has failed with error\n%s'
                    % (self.trials[i].name, tb_str)
                )
            else:
                results.append(result)
        if not results:
            self.logger.error('All %i trials have errored.' % len(self.trials))
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
    df: pd.DataFrame
    context: TrialContext

    def plot(
        self,
        config: SlopeAreaPlotConfig | None = None,
        out_fig: StrPath | None = None,
        ax: Axes | None = None,
    ) -> Axes:
        if config is None:
            config = SlopeAreaPlotConfig(hue=Column.SLOPE_TYPE)
        return slope_area_plot_single(
            data=self.df, config=config, out_fig=out_fig, ax=ax
        )


class TrialResults(UserList[TrialResult]):
    def to_dataframe(self) -> pd.DataFrame:
        return pd.concat([r.df for r in self.data])

    def plot(
        self,
        config: SlopeAreaPlotConfig | None = None,
        out_fig: StrPath | None = None,
    ) -> sns.FacetGrid:
        if config is None:
            config = SlopeAreaPlotConfig(
                hue=Column.SLOPE_TYPE, col=Column.TRIAL_NAME
            )
        return slope_area_grid(
            data=self.to_dataframe(), out_fig=out_fig, config=config
        )
