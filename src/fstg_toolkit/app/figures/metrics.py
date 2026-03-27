# Copyright 2025 ICube (University of Strasbourg - CNRS)
# author: Julien PONTABRY (ICube)
#
# This software is a computer program whose purpose is to provide a toolkit
# to model, process and analyze the longitudinal reorganization of brain
# connectivity data, as functional MRI for instance.
#
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/or redistribute the software under the terms of the CeCILL-B
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL-B license and that you accept its terms.

import math

import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from fstg_toolkit.app.figures.common import hex_to_rgba


def build_metrics_plot(metric: pd.DataFrame | pd.Series, factors: list[str]):
    if isinstance(metric, pd.Series):
        if 'Time' in metric.index.names:
            return build_longitudinal_scalar_comparison_plot(metric, factors)
        else:
            return build_scalar_comparison_plot(metric, factors)
    elif isinstance(metric, pd.DataFrame) and metric.columns.nlevels == 1:
        return build_distribution_comparison_plot(metric, factors)
    else:
        return {}


def __add_band_and_line_traces(fig: go.Figure, times: list, mean_vals: list, ci_vals: list, color: str,
                               name: str, show_legend: bool, row: int | None = None, col: int | None = None) -> None:
    """Add a 95 % confidence-interval band and a mean line to a Plotly figure.

    Three traces are appended to *fig*: an invisible upper-bound scatter, a
    lower-bound scatter filled toward the upper one (``fill='tonexty'``), and
    the mean line on top.  Hovering over the average line shows the mean and
    the 95 % CI bounds for that time point.  When *row* and *col* are provided
    the traces are placed in the corresponding subplot cell.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure (or subplot figure) to which the traces are added.
    times : list
        Sequence of x-axis values (e.g. time-point labels).
    mean_vals : list
        Group mean values, one per element of *times*.
    ci_vals : list
        Half-width of the 95 % CI at each time point
        (``1.96 × std / sqrt(n)``).  ``NaN`` entries produce no band.
    color : str
        Hex colour string (``'#RRGGBB'``) used for the mean line and the
        semi-transparent band fill.
    name : str
        Legend label for the mean line trace.
    show_legend : bool
        Whether to include this group in the figure legend.
    row : int or None, optional
        1-based subplot row index.  Must be provided together with *col*;
        ignored when ``None``.
    col : int or None, optional
        1-based subplot column index.  Must be provided together with *row*;
        ignored when ``None``.
    """
    subplot_kwargs = {}
    if row is not None and col is not None:
        subplot_kwargs = {'row': row, 'col': col}

    upper = [m + c for m, c in zip(mean_vals, ci_vals)]
    lower = [m - c for m, c in zip(mean_vals, ci_vals)]
    rgba_fill = hex_to_rgba(color, 0.2)

    hover_template = (
        'Time: %{x}<br>'
        'Average: %{y:.4f}<br>'
        '95% CI: [%{customdata[1]:.4f}, %{customdata[2]:.4f}]'
        '<extra></extra>'
    )

    fig.add_trace(
        go.Scatter(x=times, y=upper, mode='lines', line={'width': 0}, showlegend=False, hoverinfo='skip'),
        **subplot_kwargs)
    fig.add_trace(
        go.Scatter(x=times, y=lower, mode='lines', line={'width': 0}, fill='tonexty', fillcolor=rgba_fill,
                   showlegend=False, hoverinfo='skip'),
        **subplot_kwargs)
    fig.add_trace(
        go.Scatter(x=times, y=mean_vals, mode='lines', name=name, line={'color': color}, showlegend=show_legend,
                   hovertemplate=hover_template, customdata=list(zip(mean_vals, lower, upper))),
        **subplot_kwargs)


def build_longitudinal_scalar_comparison_plot(metric: pd.Series, factors: list[str]):
    """Build a longitudinal line plot with 95 % confidence-interval bands.

    For each time point the population mean and 95 % CI
    (``mean ± 1.96 × std / sqrt(n)``) are computed across subjects.  The
    result is a Plotly figure where each group defined by *factors* is drawn
    as a coloured mean line surrounded by a semi-transparent uncertainty band.

    Factor assignment follows Plotly Express conventions:

    * ``factors[0]`` → line colour
    * ``factors[1]`` → subplot rows  (optional)
    * ``factors[2]`` → subplot columns (optional)

    Parameters
    ----------
    metric : pd.Series
        Per-subject, per-time-point scalar values.  The index must contain a
        level named ``'Time'`` and may contain additional factor levels.
    factors : list of str
        Names of index levels used to split the population into groups.  Up to
        three factors are supported (colour, row, column).

    Returns
    -------
    go.Figure
        A Plotly figure with mean lines and 95 % CI bands, optionally faceted
        into a subplot grid when two or more factors are supplied.
    """
    # calculate the elements
    group = metric.groupby(['Time'] + factors)
    df = pd.DataFrame({
        metric.name: group.mean(),
        'std': group.std(),
        'n': group.count(),
    }).reset_index()
    df['ci'] = 1.96 * df['std'] / df['n'].apply(math.sqrt)

    colors = px.colors.qualitative.Plotly
    metric_name = str(metric.name)

    # when no factors, shows single plot
    if len(factors) == 0:
        fig = go.Figure()
        __add_band_and_line_traces(fig, times=df['Time'].tolist(), mean_vals=df[metric_name].tolist(),
                                   ci_vals=df['ci'].tolist(), color=colors[0], name=metric_name, show_legend=False)
        fig.update_layout(height=800, xaxis_title='Time', yaxis_title=metric_name)
        return fig

    # with factors, handle colors + faceting
    color_factor = factors[0]
    row_factor = factors[1] if len(factors) >= 2 else None
    col_factor = factors[2] if len(factors) >= 3 else None

    color_vals = df[color_factor].unique().tolist()
    row_vals = df[row_factor].unique().tolist() if row_factor else [None]
    col_vals = df[col_factor].unique().tolist() if col_factor else [None]

    if row_factor is None and col_factor is None:
        fig = go.Figure()
        for i, color_val in enumerate(color_vals):
            group_df = df[df[color_factor] == color_val]
            __add_band_and_line_traces(fig, times=group_df['Time'].tolist(), mean_vals=group_df[metric_name].tolist(),
                                       ci_vals=group_df['ci'].tolist(), color=colors[i % len(colors)], name=str(color_val),
                                       show_legend=True)
        fig.update_layout(height=800, xaxis_title='Time', yaxis_title=metric_name)
        return fig

    row_titles = [str(v) for v in row_vals] if row_factor else None
    col_titles = [str(v) for v in col_vals] if col_factor else None
    fig = make_subplots(rows=len(row_vals), cols=len(col_vals), row_titles=row_titles, column_titles=col_titles,
                        shared_xaxes=True, shared_yaxes=False)

    for r_idx, r_val in enumerate(row_vals):
        for c_idx, c_val in enumerate(col_vals):
            for i, color_val in enumerate(color_vals):
                mask = df[color_factor] == color_val
                if row_factor is not None:
                    mask &= df[row_factor] == r_val
                if col_factor is not None:
                    mask &= df[col_factor] == c_val
                group_df = df[mask]
                if group_df.empty:
                    continue
                __add_band_and_line_traces(fig, times=group_df['Time'].tolist(), mean_vals=group_df[metric_name].tolist(),
                                           ci_vals=group_df['ci'].tolist(), color=colors[i % len(colors)], name=str(color_val),
                                           show_legend=(r_idx == 0 and c_idx == 0), row=r_idx + 1, col=c_idx + 1)

    fig.update_layout(height=800)
    return fig


def build_scalar_comparison_plot(metric: pd.Series, factors: list[str]):
    params = dict(zip(('x', 'color', 'facet_row', 'facet_col'), factors))
    return px.violin(metric.reset_index(), y=metric.name, box=True, points='all', **params)


def build_distribution_comparison_plot(metric: pd.DataFrame, factors: list[str]):
    percentages = metric.divide(metric.sum(axis='columns'), axis='index') * 100

    if len(factors) > 0:
        values = percentages.groupby(factors[0]).mean().T
        labels = list(values.columns)
        x_label = factors[0]
    else:
        values = pd.DataFrame(percentages.mean())
        labels = ["all"]
        x_label = ""

    fig = go.Figure(data=[
        go.Bar(name=idx, x=labels, y=values.loc[idx], hovertemplate="%{y:.2f}%")
        for idx in values.index
    ])

    fig.update_layout(
        barmode='stack',
        yaxis_type='log',
        xaxis_title=x_label,
        yaxis_title="Percentage",
    )

    return fig
