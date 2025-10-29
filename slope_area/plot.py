from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from itertools import groupby
from pathlib import Path

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from slope_area._typing import PlotKind
from slope_area.logger import create_logger

m_logger = create_logger(__name__)


def slope_area_plot_func(
    area_col: str,
    slope_col: str,
    data: pd.DataFrame,
    config: SlopeAreaPlotConfig,
    ymin: float | None = None,
    ymax: float | None = None,
    ax: Axes | None = None,
    **plot_kwargs,
):
    if ax is not None:
        plt.sca(ax)
    area: pd.DataFrame = data[area_col]
    slope: pd.DataFrame = data[slope_col]
    slope = np.clip(slope, a_min=config.min_gradient, a_max=None)
    areamin = area.min() - 0.01
    areamax = area.max() + 1
    area_bins = 10 ** np.arange(
        np.log10(areamin),
        np.log10(areamax) + config.log_interval,
        config.log_interval,
    )
    half_bin_widths = np.append(np.diff(area_bins) / 2, 0)
    acenters = area_bins + half_bin_widths
    bin_indices = np.digitize(area, area_bins, right=True)
    data = np.column_stack([slope, area, bin_indices])
    data = data[data[:, 2].argsort()]
    if config.kind == 'line':
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
    elif config.kind == 'scatter':
        sns.scatterplot(
            x='area',
            y='slope',
            data=pd.DataFrame(data, columns=['slope', 'area', 'bin']),
            s=10,
            alpha=0.3,
            edgecolors='black',
            linewidths=0.2,
            **plot_kwargs,
        )

    if config.add_vlines:
        plt.vlines(
            area_bins,
            ymin=ymin or slope.min(),
            ymax=ymax or slope.max(),
            linewidth=0.5,
            colors='cyan',
        )
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
    hue: str | None = None
    col: str | None = None
    log_interval: float = 0.25
    min_gradient: float = 0.01
    col_wrap: int = -1
    height: int = 5
    aspect: int = 1
    title: str | None = None
    xlabel: str = 'Drainage area (m$^2$)'
    ylabel: str = 'Slope (m/m)'
    label_font_size: float = 16
    title_font_size: float = 10
    legend_font_size: float = 10
    tick_font_size: float = 14
    add_vlines: bool = False
    kind: PlotKind = 'line'
    show: bool = False


def set_plot_options(config: SlopeAreaPlotConfig, ax: Axes) -> None:
    ax.set_xlabel(config.xlabel, fontdict={'size': config.label_font_size})
    ax.set_ylabel(config.ylabel, fontdict={'size': config.label_font_size})
    ax.set_title(config.title or '', fontdict={'size': config.title_font_size})
    ax.tick_params(labelsize=config.tick_font_size)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_box_aspect(config.aspect)

    if config.hue is not None:
        ax.legend(fontsize=config.legend_font_size)

    plt.tight_layout(pad=2)


def set_plot_options_facetgrid(
    config: SlopeAreaPlotConfig, facet_grid: sns.FacetGrid
) -> None:
    facet_grid.set_xlabels(
        config.xlabel, fontdict={'size': config.label_font_size}
    )
    facet_grid.set_ylabels(
        config.ylabel, fontdict={'size': config.label_font_size}
    )
    facet_grid.set_titles(
        config.title, fontdict={'size': config.title_font_size}
    )
    facet_grid.tick_params(labelsize=config.tick_font_size)
    facet_grid.set(xscale='log', yscale='log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    if config.hue is not None:
        facet_grid.axes.flat[0].legend(fontsize=config.legend_font_size)

    plt.tight_layout(pad=2)


def slope_area_grid(
    data: pd.DataFrame,
    out_fig: Path,
    config: SlopeAreaPlotConfig | None = None,
) -> sns.FacetGrid:
    if config is None:
        config = SlopeAreaPlotConfig()
    m_logger.info('Creating slope area plot with config %s' % config)
    if config.col_wrap == -1:
        if config.col is not None:
            col_wrap = get_col_wrap(data[config.col].nunique())
        else:
            col_wrap = 1
    else:
        col_wrap = config.col_wrap
    g = sns.FacetGrid(
        data,
        hue=config.hue,
        col=config.col,
        height=config.height,
        aspect=config.aspect,
        col_wrap=col_wrap,
    )
    slope = data['slope']
    g = g.map_dataframe(
        slope_area_plot_func,
        'area',
        'slope',
        config=config,
        ymin=slope.min(),
        ymax=slope.max(),
    ).set(xscale='log', yscale='log')
    set_plot_options_facetgrid(config, g)
    plt.savefig(out_fig, dpi=300)
    m_logger.info('Saved slope area plot at %s' % out_fig)
    return g


def slope_area_plot_single(
    data: pd.DataFrame,
    out_fig: Path,
    config: SlopeAreaPlotConfig,
    ax: Axes | None = None,
) -> Axes:
    if ax is None:
        fig, ax = plt.subplots()
    func = partial(slope_area_plot_func, 'area', 'slope', config=config, ax=ax)
    if config.hue is not None:
        for grouper, group in data.groupby(config.hue):
            func(label=grouper, data=group)
    else:
        func(data=data)
    set_plot_options(config, ax)
    plt.savefig(out_fig, dpi=300)
    m_logger.info('Saved slope area plot at %s' % out_fig)
    return ax


def slope_area_plot(
    data: pd.DataFrame,
    out_fig: Path,
    config: SlopeAreaPlotConfig | None = None,
    ax: Axes | None = None,
) -> sns.FacetGrid | Axes:
    if config is None:
        config = SlopeAreaPlotConfig()
    if config.col is None:
        ret = slope_area_plot_single(data, out_fig, config, ax=ax)
    else:
        ret = slope_area_grid(data, out_fig, config)
    if config.show:
        plt.show()
    return ret
