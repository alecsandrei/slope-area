from __future__ import annotations

from slope_area.config import get_saga_env, get_wbw_env
from slope_area.enums import Column, SlopeAreaMethod, Verbose
from slope_area.features import Outlet, Outlets, Raster
from slope_area.geomorphometry import DefaultSlopeProviders
import slope_area.logger
from slope_area.logger import set_verbose
from slope_area.plot import SlopeAreaPlotConfig, slope_area_plot
from slope_area.trial import (
    AnalysisConfig,
    HydrologicAnalysisConfig,
    OutletTrialFactory,
    ResolutionTrialFactory,
    Trial,
    TrialContext,
    TrialData,
    TrialFactory,
    TrialFactoryContext,
    TrialResult,
    TrialResults,
    Trials,
    TrialsExecutor,
)

slope_area.logger.setup_logging()

__all__ = [
    # enums
    'Column',
    'SlopeAreaMethod',
    'Verbose',
    # features
    'Outlet',
    'Outlets',
    'Raster',
    # geomorphometry
    'DefaultSlopeProviders',
    # logger
    'slope_area.logger',
    'set_verbose',
    # plot
    'SlopeAreaPlotConfig',
    'slope_area_plot',
    # trial
    'AnalysisConfig',
    'HydrologicAnalysisConfig',
    'OutletTrialFactory',
    'ResolutionTrialFactory',
    'Trial',
    'TrialContext',
    'TrialData',
    'TrialFactory',
    'TrialFactoryContext',
    'TrialResult',
    'TrialResults',
    'Trials',
    'TrialsExecutor',
    # config
    'get_saga_env',
    'get_wbw_env',
]
