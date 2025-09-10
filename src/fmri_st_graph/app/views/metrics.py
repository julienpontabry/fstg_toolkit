import pandas as pd
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc

from .common import plotly_config
from ..core.io import GraphsDataset
from ..figures.metrics import build_metrics_plot

layout = [
    dbc.Row(
        dcc.Dropdown([], value='', clearable=False, id='metrics-selection')
    ),
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

    default = columns[0] if len(columns) > 0 else ''

    return columns, default


@callback(
    Output('metrics-graph', 'figure'),
    Input('metrics-selection', 'value'),
    State('store-dataset', 'data'),
    prevent_initial_call=True
)
def metric_selection_changed(selection, store_dataset):
    if store_dataset is None:
        raise PreventUpdate

    dataset = GraphsDataset.deserialize(store_dataset)
    metrics = dataset.get_metrics().reset_index()

    return build_metrics_plot(metrics, selection)
