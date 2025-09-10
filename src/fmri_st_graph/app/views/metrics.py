import pandas as pd
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc

from .common import plotly_config
from ..core.io import GraphsDataset
from ..figures.metrics import build_metrics_plot

layout = [
    dbc.Row(dcc.Dropdown([], value='', clearable=False, id='metrics-selection')),
    dbc.Row(dbc.Col(dcc.Dropdown(options=[], value=[], id='metrics-factors', multi=True, clearable=False))),
    dbc.Row(
        dcc.Loading(
            children=[dcc.Graph(figure={}, id='metrics-graph', config=plotly_config)],
            type='circle', overlay_style={"visibility": "visible", "filter": "blur(2px)"}
        )
    )
]


@callback(
    Output('metrics-selection', 'options'),
    Output('metrics-selection', 'value'),
    Output('metrics-factors', 'options'),
    Output('metrics-factors', 'value'),
    Input('store-dataset', 'data'),
    prevent_initial_call=True
)
def dataset_changed(store_dataset):
    if store_dataset is None:
        raise PreventUpdate

    dataset = GraphsDataset.deserialize(store_dataset)
    metrics = dataset.get_metrics()

    if isinstance(metrics.columns, pd.MultiIndex):
        columns = list(metrics.columns.levels[0])
    else:
        columns = list(metrics.columns)

    default_col = columns[0] if len(columns) > 0 else ''

    factors = [f"factor{i+1}" for i in range(len(dataset.factors))]
    default_factors = factors[:2]

    return columns, default_col, factors, default_factors


@callback(
    Output('metrics-graph', 'figure'),
    Input('metrics-selection', 'value'),
    Input('metrics-factors', 'value'),
    State('store-dataset', 'data'),
    prevent_initial_call=True
)
def metric_factors_selection_changed(metric_selection, factors_selection, store_dataset):
    if store_dataset is None:
        raise PreventUpdate

    dataset = GraphsDataset.deserialize(store_dataset)
    metrics = dataset.get_metrics()

    return build_metrics_plot(metrics[metric_selection], factors_selection)
