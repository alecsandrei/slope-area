from __future__ import annotations

import collections.abc as c
from itertools import groupby
from pathlib import Path
import typing as t

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if t.TYPE_CHECKING:
    from slope_area.builder import TrialResult


def slope_area_plot(
    area: np.ndarray,
    slope: np.ndarray,
    loginterval: float = 0.25,
    min_gradient: float = 0.01,
    **kwargs,
):
    slope = np.clip(slope, a_min=min_gradient, a_max=None)
    areamin = area.min() - 0.01
    areamax = area.max() + 1
    area_bins = 10 ** np.arange(
        np.log10(areamin), np.log10(areamax) + loginterval, loginterval
    )
    half_bin_widths = np.append(np.diff(area_bins) / 2, 0)
    acenters = area_bins + half_bin_widths
    bin_indices = np.digitize(area, area_bins, right=True)
    data = np.column_stack([slope, area, bin_indices])
    data = data[data[:, 2].argsort()]
    mean_slopes = []
    for bin, grouper in groupby(data, lambda arr: arr[2]):
        mean = float(np.mean([values[0] for values in grouper]))
        mean_slopes.append(mean)
    acenters_subset = acenters[np.unique(data[:, 2]).astype(int) - 1]

    sns.scatterplot(
        x='area',
        y='slope',
        data=pd.DataFrame(data, columns=['slope', 'area', 'bin']),
        s=30,
        alpha=0.2,
        color='cyan',
    )
    # plt.vlines(
    #     area_bins,
    #     ymin=slope.min(),
    #     ymax=slope.max(),
    #     linewidth=0.5,
    #     label="Intervale logaritmice",
    #     colors="blue",
    # )

    plt.plot(
        acenters_subset,
        mean_slopes,
        # color="red",
        marker='o',
        linestyle='-',
        linewidth=2,
        markersize=5,
        **kwargs,
    )
    plt.tick_params(labelsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)


def slope_area_grid(
    data: pd.DataFrame, plot_name: str, col: str, out_fig: Path
):
    g = sns.FacetGrid(
        data,
        hue='slope_type',
        col=col,
        height=5,
        aspect=1,
        col_wrap=min(data[col].unique().shape[0], 5),
    )
    g.map(slope_area_plot, 'area', 'slope').set(xscale='log', yscale='log')
    g.set_axis_labels('Drainage area (m$^2$)', 'Slope (m/m)', fontsize=16)
    g.set_titles(size=10)
    g.axes.flat[0].legend(fontsize=16, loc='lower left')
    plt.tight_layout(pad=2)
    plt.savefig(out_fig, dpi=300)
    plt.close()


def preprocess_trial_result(trial_result: TrialResult):
    slope_cols = ['Slope 3x3', 'StreamSlopeContinuous']
    df = trial_result.profiles.rename(
        columns={
            'slope_grad': 'StreamSlopeContinuous',
            'slope': 'Slope 3x3',
            'flowacc': 'area',
        }
    )
    df = df.melt(
        id_vars=df.columns.difference(slope_cols),
        value_vars=slope_cols,
        var_name='slope_type',
        value_name='values',
    )
    slope_inv = df['values'] / 100
    df['values'] = slope_inv
    df = df.rename(columns={'values': 'slope'})
    df['resolution'] = str(trial_result.resolution)
    df['name'] = trial_result.name
    return df


def preprocess_trial_results(trial_results: c.Iterable[TrialResult]):
    return pd.concat(
        [
            preprocess_trial_result(trial_result)
            for trial_result in trial_results
        ],
        ignore_index=True,
    )
