from __future__ import annotations

from slope_area import (
    Column,
    Outlet,
    SlopeAreaPlotConfig,
    Trial,
    TrialContext,
    TrialData,
    Verbose,
    set_verbose,
)
from slope_area.logger import create_logger
from slope_area.paths import PROJ_ROOT

set_verbose(Verbose.THREE)
logger = create_logger(__name__)


def main() -> None:
    logger.debug('Running __main__ script.')
    dem = PROJ_ROOT / 'data' / 'raw' / 'copdem_30m.tif'
    out_dir = PROJ_ROOT / 'data' / 'processed' / '__main__'
    out_fig = out_dir / 'slope_area.png'
    assert dem.exists()

    outlet = Outlet.from_xy(711339, 533362, name='outlet')
    context = TrialContext(
        out_dir=out_dir, data=TrialData(outlet, dem, resolution=None)
    )
    result = Trial(f'Trial {outlet.name}', context=context).run()
    result.plot(
        config=SlopeAreaPlotConfig(hue=Column.SLOPE_TYPE, legend_font_size=8),
        out_fig=out_fig,
    )


if __name__ == '__main__':
    main()
