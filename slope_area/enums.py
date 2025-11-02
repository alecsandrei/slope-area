from __future__ import annotations

from enum import StrEnum, auto


class SlopeAreaMethod(StrEnum):
    STREAMS = auto()
    MAIN_STREAM = auto()


class TrialStatus(StrEnum):
    NOT_STARTED = auto()
    RUNNING = auto()
    FINISHED = auto()
    ERRORED = auto()

    def display(self) -> str:
        return self.replace('_', ' ').capitalize()
