import networkx as nx
import numpy as np
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash_extensions.enrich import Input, Output, State, callback, dcc
from plotly import graph_objects as go


def get_measures(measure_name, g):
    if measure_name == 'Density':
        return [nx.density(g.conditional_subgraph(t=t)) for t in g.time_range]
    elif measure_name == 'Assortativity':
        return [nx.degree_assortativity_coefficient(g.conditional_subgraph(t=t)) for t in g.time_range]
    else:
        raise ValueError(f"Unknown measure: {measure_name}")


layout = [
    dbc.Row(
        dcc.Dropdown(['Density', 'Assortativity'], value='Density', clearable=False,
                     id='temporal-measure-selection')
    ),
    dbc.Row(
        dcc.Loading(
            children=[dcc.Graph(figure={}, id='temporal-measures-graph')],
            type='circle', overlay_style={"visibility": "visible", "filter": "blur(2px)"}
        )
    )
]


@callback(
    Output('temporal-measures-graph', 'figure'),
    Input('store-graphs', 'data'),
    Input('temporal-measure-selection', 'value'),
)
def update_graph_measures(graphs, measure_name):
    if graphs is None or len(graphs) == 0 or measure_name is None:
        raise PreventUpdate

    gname = next(iter(graphs))
    g = graphs[gname]
    times = list(g.time_range)

    # lines = go.Scatter(
    #     x=times,
    #     y=get_measures(measure_name, g),
    #     name=measure_name
    # )

    vals = [get_measures(measure_name, graphs[name]) for name in graphs]
    upper = np.max(vals, axis=0).tolist()
    lower = np.min(vals, axis=0).tolist()[::-1]

    ribbons = go.Scatter(
        x=times+times[::-1],
        y=upper+lower,
        fill='toself',
        name=measure_name,
    )

    return go.Figure(
        data=[ribbons],
        layout=go.Layout(
            hovermode="x",
            xaxis=dict(
                title="Time",
            ),
        ),
    )
