from __future__ import annotations

from slope_area.config import get_saga_env, get_wbw_env


def test_saga_env():
    saga_version = get_saga_env().version
    assert saga_version.major >= 7, 'Minimum SAGA GIS version should be 7.*'


def test_wbw_env():
    wbw_env = get_wbw_env()
    version = wbw_env.version
    verbose = wbw_env.verbose
    assert version is not None
    assert not verbose
