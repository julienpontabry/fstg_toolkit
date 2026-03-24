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

import json

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc, html, no_update
from dash.exceptions import PreventUpdate
from plotly import graph_objects as go
from plotly.io import to_json

from .common import (
    plotly_config,
    build_factors_options,
    create_factors_options_controls,
)
from ..figures.frequent import build_pattern_figure, build_pattern_frequency_plot
from ...frequent import PatternEquivalenceStrategyRegistry
from ...io import GraphsDataset

EQUIVALENCE_OPTIONS = PatternEquivalenceStrategyRegistry.names()
ANALYSIS_OPTIONS = ['Pattern distribution', 'Occurrence distribution']


layout = [
    dbc.Row([
        dbc.Col(dbc.Label("Equivalence class"), width='auto'),
        dbc.Col(dcc.Dropdown(EQUIVALENCE_OPTIONS, value='structure-transitions', clearable=False, id='frequent-equivalence'))
    ]),
    dbc.Row(dcc.Dropdown([], value='', clearable=False, id='frequent-mode',
                         style={'display': 'none'})),
    dbc.Row(dcc.Dropdown([], value='', clearable=False, id='frequent-analysis')),
    dbc.Row(dbc.Col(create_factors_options_controls('frequent'))),
    dbc.Row(html.Div([
        dcc.Loading(
            children=[dcc.Graph(figure={}, id='frequent-graph', config=plotly_config, clear_on_unhover=True)],
            type='circle', overlay_style={'visibility': 'visible', 'filter': 'blur(2px)'}
        ),
        dcc.Tooltip(id='frequent-pattern-tooltip', style={'max-width': '320px', 'padding': '0'})
    ], style={'position': 'relative'})),  # div + relative position needed to keep tooltip at the right place
    dcc.Store(id='frequent-patterns-store'),
]


@callback(
    Output('frequent-mode', 'options'),
    Output('frequent-mode', 'value'),
    Output('frequent-factors', 'options'),
    Output('frequent-factors', 'value'),
    Input('store-dataset', 'data')
)
def dataset_changed(store_dataset: dict) -> tuple:
    if store_dataset is None:
        raise PreventUpdate

    dataset = GraphsDataset.deserialize(store_dataset)

    modes = dataset.get_available_frequent_pattern_modes()
    default_mode = modes[0] if len(modes) > 0 else ''

    factors, default_factors = build_factors_options(dataset)

    return modes, default_mode, factors, default_factors


@callback(
    Output('frequent-analysis', 'options'),
    Output('frequent-analysis', 'value'),
    Input('frequent-mode', 'value'),
    prevent_initial_call=True
)
def mode_changed(mode: str) -> tuple:
    if not mode:
        raise PreventUpdate

    return ANALYSIS_OPTIONS, ANALYSIS_OPTIONS[0]


@callback(
    Output('frequent-graph', 'figure'),
    Output('frequent-patterns-store', 'data'),
    Input('frequent-analysis', 'value'),
    Input('frequent-equivalence', 'value'),
    Input('frequent-factors', 'value'),
    State('store-dataset', 'data'),
    State('frequent-mode', 'value'),
    prevent_initial_call=True,
)
def analysis_selection_changed(analysis: str, equivalence_strategy: str, factors_selection: list[str],
                               store_dataset: dict, mode: str) -> tuple:
    if store_dataset is None or not mode or not analysis:
        raise PreventUpdate

    dataset = GraphsDataset.deserialize(store_dataset)
    equivalence_strategy = PatternEquivalenceStrategyRegistry.get(equivalence_strategy)
    analysis = dataset.get_frequent_patterns_analysis(mode, equivalence_strategy)

    pattern_figures = [to_json(build_pattern_figure(p)) for p in analysis.unique_patterns]

    return build_pattern_frequency_plot(analysis, factors_selection), pattern_figures


@callback(
    Output('frequent-pattern-tooltip', 'show'),
    Output('frequent-pattern-tooltip', 'bbox'),
    Output('frequent-pattern-tooltip', 'children'),
    Input('frequent-graph', 'hoverData'),
    State('frequent-patterns-store', 'data'),
    prevent_initial_call=True,
)
def show_pattern_tooltip(hover_data: dict, patterns_json: list) -> tuple:
    if hover_data is None or patterns_json is None:
        return False, no_update, no_update

    point = hover_data['points'][0]
    pattern_index = int(point['x']) - 1  # histogram x is 1-based

    if pattern_index < 0 or pattern_index >= len(patterns_json):
        return False, no_update, no_update

    fig = go.Figure(json.loads(patterns_json[pattern_index]))
    bbox = point['bbox']

    return True, bbox, dcc.Graph(figure=fig, config={'displayModeBar': False},
                                 style={'width': '200px', 'height': '200px'})
