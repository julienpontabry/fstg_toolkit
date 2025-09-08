import networkx as nx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc
from plotly import graph_objects as go

from ..core.io import GraphsDataset


layout = [
    dbc.Row(
        dcc.Dropdown([], value='', clearable=False, id='metrics-selection')
    ),
    dbc.Row(
        dcc.Loading(
            children=[dcc.Graph(figure={}, id='temporal-measures-graph')],
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
    columns = list(metrics.columns)
    default = columns[0] if len(columns) > 0 else ''

    return columns, default
