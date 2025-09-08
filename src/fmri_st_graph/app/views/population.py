import networkx as nx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback, dcc
from plotly import graph_objects as go

from ..core.io import GraphsDataset


def get_measures(measure_name, g):
    if measure_name == 'Density':
        return [nx.density(g.sub(t=t)) for t in g.time_range]
    elif measure_name == 'Assortativity':
        return [nx.degree_assortativity_coefficient(g.sub(t=t)) for t in g.time_range]
    else:
        raise ValueError(f"Unknown measure: {measure_name}")


layout = [
    dbc.Row(
        dcc.Dropdown(['Density', 'Assortativity'], value='Density', clearable=False,
                     id='metrics-selection')
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
