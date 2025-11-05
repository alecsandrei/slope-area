from __future__ import annotations

from enum import IntEnum, StrEnum, auto


class Verbose(IntEnum):
    ZERO = 0  # no stdout, no stderr
    ONE = 1  # stderr (logging level WARNING)
    TWO = 2  # stderr + stdout (logging level INFO)
    THREE = 3  # stderr + stdout (logging level DEBUG)


class SlopeAreaMethod(StrEnum):
    # Slope area values for all of the streams
    STREAMS = auto()

    # Slope area values for the main stream
    MAIN_STREAM = auto()


class TrialStatus(StrEnum):
    NOT_STARTED = auto()
    RUNNING = auto()
    FINISHED = auto()
    ERRORED = auto()


class Column(StrEnum):
    AREA_VALUES = 'area'
    SLOPE_VALUES = 'slope'
    SLOPE_TYPE = 'slope_type'
    TRIAL_NAME = 'trial'
    RESOLUTION = 'resolution'
