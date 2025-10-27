from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from slope_area._typing import PlotKind
from slope_area.logger import create_logger

m_logger = create_logger(__name__)


def slope_area_plot(
    area: np.ndarray,
    slope: np.ndarray,
    min_gradient: float = 0.01,
    log_interval: float = 0.25,
    add_vlines: bool = False,
    add_scatter: bool = False,
    ymin: float | None = None,
    ymax: float | None = None,
    kind: PlotKind = 'line',
    **plot_kwargs,
):
    slope = np.clip(slope, a_min=min_gradient, a_max=None)
    areamin = area.min() - 0.01
    areamax = area.max() + 1
    area_bins = 10 ** np.arange(
        np.log10(areamin), np.log10(areamax) + log_interval, log_interval
    )
    half_bin_widths = np.append(np.diff(area_bins) / 2, 0)
    acenters = area_bins + half_bin_widths
    bin_indices = np.digitize(area, area_bins, right=True)
    data = np.column_stack([slope, area, bin_indices])
    data = data[data[:, 2].argsort()]
    if kind == 'line':
        mean_slopes = []
        for bin, grouper in groupby(data, lambda arr: arr[2]):
            mean = float(np.mean([values[0] for values in grouper]))
            mean_slopes.append(mean)
        acenters_subset = acenters[np.unique(data[:, 2]).astype(int) - 1]
        plt.plot(
            acenters_subset,
            mean_slopes,
            marker='o',
            linestyle='-',
            linewidth=2,
            markersize=5,
            **plot_kwargs,
        )
    elif kind == 'scatter':
        sns.scatterplot(
            x='area',
            y='slope',
            data=pd.DataFrame(data, columns=['slope', 'area', 'bin']),
            s=10,
            alpha=0.1,
            edgecolors='black',
            linewidths=0.2,
            **plot_kwargs,
        )

    if add_vlines:
        plt.vlines(
            area_bins,
            ymin=ymin or slope.min(),
            ymax=ymax or slope.max(),
            linewidth=0.5,
            colors='cyan',
        )
    plt.tick_params(labelsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)


def get_col_wrap(n_unique: int):
    for wrap in range(5, 2, -1):
        if n_unique % wrap == 0:
            col_wrap = wrap
            break
    else:
        col_wrap = min(n_unique, 5)
    m_logger.info('Infered %i cols for the plot' % col_wrap)
    return col_wrap


@dataclass(kw_only=True)
class SlopeAreaPlotConfig:
    log_interval: float = 0.25
    min_gradient: float = 0.01
    col_wrap: int = -1
    height: int = 5
    aspect: int = 1
    add_vlines: bool = False
    kind: PlotKind = 'line'
    show: bool = False


def slope_area_grid(
    data: pd.DataFrame, col: str, out_fig: Path, config: SlopeAreaPlotConfig
):
    m_logger.info('Creating slope area plot with config %s' % config)
    if config.col_wrap == -1:
        col_wrap = get_col_wrap(data[col].nunique())
    else:
        col_wrap = config.col_wrap
    g = sns.FacetGrid(
        data,
        hue='slope_type',
        col=col,
        height=config.height,
        aspect=config.aspect,
        col_wrap=col_wrap,
    )
    slope = data['slope']
    g = g.map(
        slope_area_plot,
        'area',
        'slope',
        min_gradient=config.min_gradient,
        log_interval=config.log_interval,
        add_vlines=config.add_vlines,
        kind=config.kind,
        ymin=slope.min(),
        ymax=slope.max(),
    ).set(xscale='log', yscale='log')
    g.set_axis_labels('Drainage area (m$^2$)', 'Slope (m/m)', fontsize=16)
    g.set_titles(size=10)
    g.axes.flat[0].legend(fontsize=10)
    plt.tight_layout(pad=2)
    plt.savefig(out_fig, dpi=300)
    m_logger.info('Saved slope area plot at %s' % out_fig)
    if config.show:
        plt.show()
    plt.close()
