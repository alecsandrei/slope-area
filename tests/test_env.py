from __future__ import annotations

from slope_area import SAGA_ENV, WBW_ENV


def test_saga_env():
    saga_version = SAGA_ENV.version
    assert saga_version.major >= 7, 'Minimum SAGA GIS version should be 7.*'


def test_wbw_env():
    version = WBW_ENV.version
    verbose = WBW_ENV.verbose
    assert version is not None
    assert not verbose
