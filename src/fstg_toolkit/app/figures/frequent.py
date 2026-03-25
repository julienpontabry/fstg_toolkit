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
from typing import Any, Callable, Optional

import networkx as nx
import plotly.express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from fstg_toolkit.app.figures.common import integer_tick_step
from fstg_toolkit.frequent import FrequentPattern, FrequentPatternsPopulationAnalysis

FrequentAnalysisBuilder = Callable[[FrequentPatternsPopulationAnalysis, list[str]], go.Figure]


class FrequentFigureBuilderRegistry:
    """Registry for frequent pattern analysis figure builders.

    Builders self-register via the ``@FrequentFigureBuilderRegistry.register(name)``
    decorator.  Look up by the registered display-name string.

    Each builder may optionally declare compatible modes (e.g. ``{'s', 'st'}``)
    and a tooltip type.  A mode of ``None`` means the builder is available in
    all modes.  A tooltip of ``None`` means no custom tooltip is shown.

    Tooltip types
    -------------
    ``None``
        No custom tooltip.
    ``'pattern'``
        Hover point carries a 1-based pattern index on *x*; show the
        corresponding pattern graph and its count.
    ``'pattern-pair'``
        Hover point carries 1-based pattern indices on both *x* and *y*
        (heatmap cell); show both pattern graphs and the co-occurrence count.
    """

    _analyses: dict[str, tuple[FrequentAnalysisBuilder, Optional[set[str]], Optional[str], Optional[str]]] = {}

    @classmethod
    def register(cls, name: str, modes: Optional[set[str]] = None,
                 tooltip: Optional[str] = None,
                 description: Optional[str] = None) -> Callable[[FrequentAnalysisBuilder], FrequentAnalysisBuilder]:
        """Class decorator factory that registers a builder under the given name.

        Parameters
        ----------
        name : str
            The display name to register the builder under.
        modes : set[str] or None, optional
            Set of compatible modes (e.g. ``{'s', 'st'}``).  ``None`` means
            the builder is available in all modes.
        tooltip : str or None, optional
            Tooltip type for this figure.  One of ``'pattern'``,
            ``'pattern-pair'``, or ``None`` (no tooltip).
        description : str or None, optional
            Short human-readable description of what the figure shows.

        Returns
        -------
        Callable
            Decorator that stores the builder and returns it unchanged.
        """
        def decorator(builder: FrequentAnalysisBuilder) -> FrequentAnalysisBuilder:
            cls._analyses[name] = (builder, modes, tooltip, description)
            return builder
        return decorator

    @classmethod
    def get(cls, name: str) -> FrequentAnalysisBuilder:
        """Look up a builder by its registered name.

        Parameters
        ----------
        name : str
            The registered display name.

        Returns
        -------
        FrequentAnalysisBuilder
            The registered figure builder callable.

        Raises
        ------
        KeyError
            If no builder is registered under this name.
        """
        return cls._analyses[name][0]

    @classmethod
    def tooltip_type(cls, name: str) -> Optional[str]:
        """Return the tooltip type declared for the given figure builder.

        Parameters
        ----------
        name : str
            The registered display name.

        Returns
        -------
        str or None
            The tooltip type (``'pattern'``, ``'pattern-pair'``, or ``None``).

        Raises
        ------
        KeyError
            If no builder is registered under this name.
        """
        return cls._analyses[name][2]

    @classmethod
    def get_description(cls, name: str) -> Optional[str]:
        """Return the description declared for the given figure builder.

        Parameters
        ----------
        name : str
            The registered display name.

        Returns
        -------
        str or None
            The human-readable description, or ``None`` if not set.

        Raises
        ------
        KeyError
            If no builder is registered under this name.
        """
        return cls._analyses[name][3]

    @classmethod
    def names(cls, mode: Optional[str] = None) -> list[str]:
        """Return the names of builders compatible with the given mode.

        Parameters
        ----------
        mode : str
            The current mode (e.g. ``'s'``, ``'t'``, or ``'st'``).

        Returns
        -------
        list[str]
            Sorted list of compatible builder display names.
        """
        return sorted(
            name for name, (_, modes, _tooltip, _desc) in cls._analyses.items()
            if modes is None or mode is None or mode in modes
        )


def build_pattern_figure(pattern: FrequentPattern) -> go.Figure:
    """Build a small Plotly figure visualizing a single frequent pattern graph.

    Parameters
    ----------
    pattern : FrequentPattern
        The frequent pattern to visualize.

    Returns
    -------
    go.Figure
        A Plotly figure showing the pattern as a directed graph.
    """
    if len(pattern) > 2:
        initial_pos = nx.spectral_layout(pattern)
        pos = nx.spring_layout(pattern, pos=initial_pos, iterations=20, seed=42)
    else:
        pos = nx.spring_layout(pattern, iterations=20, seed=42)

    fig = go.Figure()

    # Draw edges as lines
    for u, v, data in pattern.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines',
            line={'color': 'gray', 'width': 1.5},
            hoverinfo='skip',
            showlegend=False,
        ))

    # Draw arrowheads via annotations
    for u, v in pattern.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        # Shorten arrow slightly so head doesn't overlap node marker
        dx, dy = x1 - x0, y1 - y0
        length = math.sqrt(dx * dx + dy * dy)
        if length > 0:
            shrink = min(0.08, length * 0.15)
            x1_s = x1 - (dx / length) * shrink
            y1_s = y1 - (dy / length) * shrink
        else:
            x1_s, y1_s = x1, y1
        fig.add_annotation(
            x=x1_s, y=y1_s, ax=x0, ay=y0,
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True,
            arrowhead=3, arrowsize=1.0, arrowwidth=1.5,
            arrowcolor='gray',
            standoff=0,
        )

    # Draw nodes
    node_x = [pos[n][0] for n in pattern.nodes()]
    node_y = [pos[n][1] for n in pattern.nodes()]

    regions = [pattern.nodes[n].get('region', '') for n in pattern.nodes()]
    all_same = len(set(regions)) <= 1
    node_labels = ['' if all_same else r for r in regions]

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker={'size': 15, 'color': 'lightblue', 'line': {'width': 1.5, 'color': 'steelblue'}},
        text=node_labels,
        textposition='middle center',
        textfont={'size': 9},
        hoverinfo='skip',
        showlegend=False,
    ))

    # Draw edge labels at midpoints
    for u, v, data in pattern.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        label = data.get('transition', '')
        fig.add_trace(go.Scatter(
            x=[(x0 + x1) / 2], y=[(y0 + y1) / 2],
            mode='text',
            text=[str(label)],
            textfont={'size': 10, 'color': 'darkred'},
            hoverinfo='skip',
            showlegend=False,
        ))

    fig.update_layout(
        width=200, height=200,
        xaxis={'visible': False}, yaxis={'visible': False},
        margin={'l': 10, 'r': 10, 't': 10, 'b': 10},
        plot_bgcolor='white',
    )

    return fig


def __build_faceted_heatmap(data_by_group: dict[tuple[str, ...], tuple[list[str], list[list[int]]]],
                           factors: list[str], axis_title: str) -> go.Figure:
    """Build a heatmap figure with one subplot per factor group.

    Parameters
    ----------
    data_by_group : dict[tuple[str, ...], tuple[list[str], list[list[int]]]]
        Mapping from factor-group tuples to ``(labels, symmetric_2d_matrix)``.
    factors : list[str]
        Factor names used for subplot titles.
    axis_title : str
        Label for both axes.

    Returns
    -------
    go.Figure
        A Plotly figure with one heatmap per factor group.
    """
    groups = sorted(data_by_group.keys())

    # Compute grid dimensions for rows/columns faceting
    if len(factors) == 0:
        row_vals: list[Any] = [()]
        col_vals: list[Any] = [()]
    elif len(factors) == 1:
        row_vals = groups
        col_vals = [('',)]
    else:
        row_vals = sorted({g[0] for g in groups})
        col_vals = sorted({g[1] for g in groups})

    n_rows = len(row_vals)
    n_cols = len(col_vals)

    # Map each group key to its 1-based (row, col) position
    if len(factors) == 0:
        group_to_rc: dict[tuple[str, ...], tuple[int, int]] = {(): (1, 1)}
    elif len(factors) == 1:
        group_to_rc = {key: (r, 1) for r, key in enumerate(row_vals, start=1)}
    else:
        row_index = {v: i for i, v in enumerate(row_vals, start=1)}
        col_index = {v: i for i, v in enumerate(col_vals, start=1)}
        group_to_rc = {key: (row_index[key[0]], col_index[key[1]]) for key in groups}

    # Subplot titles in row-major order (required by make_subplots)
    if len(factors) == 0:
        subplot_titles: list[str] = ['']
    elif len(factors) == 1:
        subplot_titles = [f'{factors[0]}={rv[0]}' for rv in row_vals]
    else:
        subplot_titles = [
            f'{factors[0]}={rv} / {factors[1]}={cv}'
            for rv in row_vals
            for cv in col_vals
        ]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.13,
    )

    last_key = groups[-1]

    for group_key in groups:
        row, col = group_to_rc[group_key]
        labels, matrix = data_by_group[group_key]
        fig.add_trace(
            go.Heatmap(
                z=matrix, x=labels, y=labels,
                colorscale='Blues',
                showscale=(group_key == last_key),
            ),
            row=row, col=col,
        )
        fig.update_xaxes(title_text=axis_title, row=row, col=col)
        fig.update_yaxes(title_text=axis_title, row=row, col=col)

    fig.update_layout(height=max(800, 500 * n_rows))

    return fig


@FrequentFigureBuilderRegistry.register('Patterns distribution', tooltip='pattern',
                                        description='Distribution of unique patterns. Helps identify most prevalent '
                                                    'patterns and distribution differences among groups.')
def build_pattern_frequency_plot(analysis: FrequentPatternsPopulationAnalysis, factors: list[str]) -> go.Figure:
    counts = analysis.get_counts(factors)

    # add 1 to pattern indices to start from 1
    # (and last is the total number of patterns)
    counts = counts.reset_index()
    counts['idx'] += 1

    params = dict(zip(('facet_row', 'facet_col'), factors))
    fig = px.bar(
        counts, x='idx', y='Count', **params,
        barmode='group',
        labels={'idx': 'Pattern'},
        height=800,
    )

    # hide the default bar tooltip (custom pattern tooltip is used instead)
    fig.update_traces(hoverinfo='none', hovertemplate=None)

    # only integer ticks
    fig.update_yaxes(tick0=0, dtick=integer_tick_step(int(counts['Count'].max())))

    return fig


@FrequentFigureBuilderRegistry.register('Temporal dynamics per region', modes={'t', 'st'},
                                        description='Distribution of patterns count among region and temporal transitions. '
                                                    'Reveals how different regions participate in temporal organization of the system.')
def build_temporal_dynamics_plot(analysis: FrequentPatternsPopulationAnalysis, factors: list[str]) -> go.Figure:
    """Stacked bar chart of RC5 transition types per brain region.

    Parameters
    ----------
    analysis : FrequentPatternsPopulationAnalysis
        The population analysis to visualize.
    factors : list[str]
        Factor columns to facet by.

    Returns
    -------
    go.Figure
        A Plotly figure with regions on x-axis and transition type counts as stacked bars.
    """
    df = analysis.get_temporal_dynamics(factors)

    params = dict(zip(('facet_row', 'facet_col'), factors))
    fig = px.bar(
        df, x='Region', y='Count', color='Transition', **params,
        custom_data=['PatternIndices'],
        barmode='stack',
        height=800,
    )

    # only integer ticks
    fig.update_yaxes(tick0=0, dtick=integer_tick_step(int(df['Count'].max())))

    fig.update_traces(hovertemplate=(
        'Region: %{x}<br>Transition: %{data.name}<br>Count: %{y}'
        '<br>Patterns: %{customdata[0]}<extra></extra>'
    ))

    return fig


@FrequentFigureBuilderRegistry.register('Region co-occurrence', modes={'s', 'st'},
                                        description='Heatmap showing how often two region appear together in frequent patterns. '
                                                    'Reveals structural associations and identifies strongly connected regions.')
def build_region_co_occurrence_plot(analysis: FrequentPatternsPopulationAnalysis, factors: list[str]) -> go.Figure:
    """Symmetric heatmap of region co-occurrence via spatial edges.

    Parameters
    ----------
    analysis : FrequentPatternsPopulationAnalysis
        The population analysis to visualize.
    factors : list[str]
        Factor columns to facet by.

    Returns
    -------
    go.Figure
        A Plotly heatmap with region pairs as axes.
    """
    data = analysis.get_region_co_occurrence(factors)
    return __build_faceted_heatmap(data, factors, 'Region')


@FrequentFigureBuilderRegistry.register('Patterns per region',
                                        description='Distribution of patterns count among the regions. '
                                                    'Identifies most dynamic regions.')
def build_patterns_per_region_plot(analysis: FrequentPatternsPopulationAnalysis, factors: list[str]) -> go.Figure:
    """Bar chart of pattern counts per brain region.

    Parameters
    ----------
    analysis : FrequentPatternsPopulationAnalysis
        The population analysis to visualize.
    factors : list[str]
        Factor columns to facet by.

    Returns
    -------
    go.Figure
        A Plotly bar chart with regions on x-axis and pattern counts on y-axis.
    """
    df = analysis.get_patterns_per_region(factors)

    params = dict(zip(('facet_row', 'facet_col'), factors))
    fig = px.bar(
        df, x='Region', y='Count', **params,
        custom_data=['PatternIndices'],
        barmode='group',
        height=800,
    )

    # only integer ticks
    fig.update_yaxes(tick0=0, dtick=integer_tick_step(int(df['Count'].max())))

    fig.update_traces(hovertemplate=(
        'Region: %{x}<br>Count: %{y}<br>Patterns: %{customdata[0]}<extra></extra>'
    ))

    return fig


@FrequentFigureBuilderRegistry.register('Pattern co-occurrence', tooltip='pattern-pair',
                                        description='Heatmap of the number of subjects that simultaneously exhibit both patterns. '
                                                    'Reveals which pattern combinations tend to co-occur, indicating related or dependent processes.')
def build_pattern_co_occurrence_plot(analysis: FrequentPatternsPopulationAnalysis, factors: list[str]) -> go.Figure:
    """Symmetric heatmap of pattern co-occurrence across subjects.

    Parameters
    ----------
    analysis : FrequentPatternsPopulationAnalysis
        The population analysis to visualize.
    factors : list[str]
        Factor columns to facet by.

    Returns
    -------
    go.Figure
        A Plotly heatmap where cell (i, j) = number of subjects with both patterns.
    """
    data = analysis.get_pattern_cooccurrence(factors)

    n = len(analysis.unique_patterns)
    labels = [str(i + 1) for i in range(n)]  # 1-indexed pattern labels
    heatmap_data = {key: (labels, matrix) for key, matrix in data.items()}
    fig = __build_faceted_heatmap(heatmap_data, factors, 'Pattern')

    # hide default tooltip so only the custom pattern-pair tooltip is shown
    fig.update_traces(hoverinfo='none', hovertemplate=None)

    return fig


@FrequentFigureBuilderRegistry.register('Occurrence histogram',
                                        description='Histogram of unique pattern occurrences. Shows the prevalence spectrum from unique to highly common patterns.')
def build_occurrence_histogram_plot(analysis: FrequentPatternsPopulationAnalysis, factors: list[str]) -> go.Figure:
    """Histogram of pattern occurrence counts.

    Parameters
    ----------
    analysis : FrequentPatternsPopulationAnalysis
        The population analysis to visualize.
    factors : list[str]
        Factor columns to facet by.

    Returns
    -------
    go.Figure
        A Plotly bar chart with occurrence counts on x-axis and number of patterns on y-axis.
    """
    df = analysis.get_occurrence_histogram(factors)

    params = dict(zip(('facet_row', 'facet_col'), factors))
    fig = px.bar(
        df, x='Occurrences', y='Patterns', **params,
        custom_data=['PatternIndices'],
        barmode='group',
        height=800,
        labels={'Patterns': 'Patterns count'},
    )

    # only integer ticks
    fig.update_xaxes(tick0=0, dtick=integer_tick_step(int(df['Occurrences'].max())))
    fig.update_yaxes(tick0=0, dtick=integer_tick_step(int(df['Patterns'].max())))

    fig.update_traces(hovertemplate=(
        'Occurrences: %{x}<br>Patterns count: %{y}<br>Patterns: %{customdata[0]}<extra></extra>'
    ))

    return fig


@FrequentFigureBuilderRegistry.register('Pattern size',
                                        description='Distribution of patterns by their number of nodes (graph size). Characterizes the complexity and scale of discovered patterns.')
def build_pattern_complexity_plot(analysis: FrequentPatternsPopulationAnalysis, factors: list[str]) -> go.Figure:
    """Histogram of pattern sizes (node counts).

    Parameters
    ----------
    analysis : FrequentPatternsPopulationAnalysis
        The population analysis to visualize.
    factors : list[str]
        Factor columns to facet by.

    Returns
    -------
    go.Figure
        A Plotly bar chart with pattern size on x-axis and count on y-axis.
    """
    df = analysis.get_pattern_complexity(factors)

    params = dict(zip(('facet_row', 'facet_col'), factors))
    fig = px.bar(
        df, x='Size', y='Count', **params,
        custom_data=['PatternIndices'],
        barmode='group',
        labels={'Size': 'Pattern size (nodes)'},
        height=800,
    )

    # only integer ticks
    fig.update_xaxes(tick0=0, dtick=integer_tick_step(int(df['Size'].max())))
    fig.update_yaxes(tick0=0, dtick=integer_tick_step(int(df['Count'].max())))

    fig.update_traces(hovertemplate=(
        'Size: %{x}<br>Count: %{y}<br>Patterns: %{customdata[0]}<extra></extra>'
    ))

    return fig
