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

import networkx as nx
import plotly.express as px
from plotly import graph_objects as go

from fstg_toolkit.frequent import FrequentPattern, FrequentPatternsPopulationAnalysis


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

    # force integer scale on y-axis
    fig.update_yaxes(dtick=1)

    return fig
