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

import pandas as pd
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc

from .common import plotly_config
from ..core.io import GraphsDataset
from ..figures.metrics import build_metrics_plot

layout = [
    dbc.Row(dcc.Dropdown([], value='', clearable=False, id='metrics-type')),
    dbc.Row(dcc.Dropdown([], value='', clearable=False, id='metrics-selection')),
    dbc.Row(dbc.Col(dcc.Dropdown(options=[], value=[], id='metrics-factors', multi=True, clearable=False))),
    dbc.Row(
        dcc.Loading(
            children=[dcc.Graph(figure={}, id='metrics-graph', config=plotly_config)],
            type='circle', overlay_style={'visibility': 'visible', 'filter': 'blur(2px)'}
        )
    )
]


@callback(
    Output('metrics-type', 'options'),
    Output('metrics-type', 'value'),
    Output('metrics-factors', 'options'),
    Output('metrics-factors', 'value'),
    Input('store-dataset', 'data'),
    prevent_initial_call=True
)
def dataset_changed(store_dataset):
    if store_dataset is None:
        raise PreventUpdate

    dataset = GraphsDataset.deserialize(store_dataset)

    metrics_types = [t.capitalize() for t in dataset.get_available_metrics()]
    default_metrics_type = metrics_types[0] if len(metrics_types) > 0 else ''

    factors = [f"Factor{i+1}" for i in range(len(dataset.factors))]
    default_factors = factors[:2]

    return metrics_types, default_metrics_type, factors, default_factors


@callback(
Output('metrics-selection', 'options'),
    Output('metrics-selection', 'value'),
    Input('metrics-type', 'value'),
    State('store-dataset', 'data'),
    prevent_initial_call=True
)
def metrics_type_changed(metrics_type, store_dataset):
    if store_dataset is None or metrics_type == '':
        raise PreventUpdate

    dataset = GraphsDataset.deserialize(store_dataset)
    metrics = dataset.get_metrics(metrics_type.lower())

    if isinstance(metrics.columns, pd.MultiIndex):
        metric_names = list(metrics.columns.levels[0])
    else:
        metric_names = list(metrics.columns)

    default_name = metric_names[0] if len(metric_names) > 0 else ""

    return metric_names, default_name


@callback(
    Output('metrics-graph', 'figure'),
    Input('metrics-selection', 'value'),
    Input('metrics-factors', 'value'),
    State('store-dataset', 'data'),
    State('metrics-type', 'value'),
    prevent_initial_call=True,
)
def metric_factors_selection_changed(metric_selection, factors_selection, store_dataset, metrics_type):
    if store_dataset is None and metrics_type == '':
        raise PreventUpdate

    dataset = GraphsDataset.deserialize(store_dataset)
    metrics = dataset.get_metrics(metrics_type.lower())

    return build_metrics_plot(metrics[metric_selection], factors_selection)
