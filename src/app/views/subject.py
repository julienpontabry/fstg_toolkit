from typing import Any

import numpy as np
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash_extensions.enrich import Input, Output, State, callback, dcc
from fmri_st_graph.graph import RC5
from fmri_st_graph.visualization import __CoordinatesGenerator, _trans_color
from plotly import graph_objects as go


def generate_subject_display_props(graph, name: str, regions: list[str]) -> dict[str, Any]:
    start_graph = graph.conditional_subgraph(t=0)

    # define nodes' properties
    nodes_coord_gen = __CoordinatesGenerator(graph)
    nodes_coord = {}
    nodes_x = []
    nodes_y = []
    nodes_color = []
    nodes_sizes = []
    levels = [0]

    for region in regions:
        nodes = [n for n, d in start_graph.nodes.items() if d['region'] == region]
        coord = nodes_coord_gen.generate(nodes, levels[-1])
        x, y = tuple(zip(*coord.values()))
        nodes_color += [graph.nodes[n]['internal_strength'] for n in coord.keys()]
        nodes_sizes += [graph.nodes[n]['efficiency'] for n in coord.keys()]

        levels.append(max(y) + 2)

        nodes_coord |= coord
        nodes_x += x
        nodes_y += y

    # define edges' properties
    edges_x = []
    edges_y = []
    edges_colors = []
    for n in nodes_coord:
        for m, d in graph[n].items():
            if d['type'] == 'temporal' and d['transition'] != RC5.EQ:
                edges_x.append((nodes_coord[n][0], nodes_coord[m][0]))
                edges_y.append((nodes_coord[n][1], nodes_coord[m][1]))
                edges_colors.append(_trans_color(d['transition']))

    return {'nodes_x': nodes_x, 'nodes_y': nodes_y,
            'nodes_color': nodes_color, 'nodes_sizes': nodes_sizes,
            'edges_x': edges_x, 'edges_y': edges_y, 'edges_colors': edges_colors,
            'levels': levels, 'height': levels[-1] - 2, 'regions': regions}


def build_subject_figure(graph, name: str, regions: list[str]) -> go.Figure:
    props = generate_subject_display_props(graph, name, regions)

    # TODO find a way to reduce the number of elements displayed in the figure
    nodes_trace = go.Scatter(
        x=props['nodes_x'], y=props['nodes_y'],
        mode='markers',
        hoverinfo='none',
        marker=dict(size=6*np.power(props['nodes_sizes'], 5),
                    color=props['nodes_color'],
                    cmin=-1, cmid=0, cmax=1, line_width=0,
                    colorscale='RdBu_r', showscale=True)
    )

    edges_traces = []
    for x, y, c in zip(props['edges_x'], props['edges_y'], props['edges_colors']):
        edges_trace = go.Scatter(
            x=x, y=y,
            mode='lines',
            hoverinfo='skip',
            line=dict(width=0.5, color=c)
        )
        edges_traces.append(edges_trace)

    return go.Figure(
        data=[*edges_traces, nodes_trace],
        layout=go.Layout(
            height=21*(props['height']+2)+126,
            margin=dict(t=40),
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, title="Time",
                       showspikes=True, spikemode='across', spikesnap='cursor',
                       spikedash='solid', spikecolor='black', spikethickness=0.5),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=True,
                       tickvals=props['levels'][:-1], ticktext=props['regions'],
                       minor=dict(tickvals=np.subtract(props['levels'][1:-1], 1),
                                  showgrid=True, gridwidth=2, griddash='dash',
                                  gridcolor='lightgray')
                       )
        )
    )


layout = [
    dbc.Row(
        dcc.Dropdown([], clearable=False, id='subject-selection')
    ),
    dbc.Row([
        dbc.Col(dcc.Dropdown([], multi=True, placeholder="Select regions...", id='regions-selection'), width=11),
        dbc.Col(dbc.Button("Apply", color='secondary', id='apply-button'),
                className='d-grid gap-2 d-md-block', align='center')
    ], className='g-0'),
    dbc.Row(
        dcc.Loading(
            children=[dcc.Graph(figure={}, id='st-graph')],
            type='circle', overlay_style={"visibility": "visible", "filter": "blur(2px)"}
        )
    )
]


@callback(
    Output('regions-selection', 'options'),
    Output('regions-selection', 'value'),
    Input('store-desc', 'data'),
)
def update_regions(desc):
    if desc is None:
        raise PreventUpdate

    regions = desc.sort_values("Name_Region")["Name_Region"].unique().tolist()
    return regions, regions


@callback(
Output('subject-selection', 'options'),
    Output('subject-selection', 'value'),
    Input('store-corr', 'data'),
)
def update_subjects(corr):
    if corr is None or len(corr) == 0:
        raise PreventUpdate

    return list(corr.keys()), next(iter(corr.keys()))


@callback(
    Output('st-graph', 'figure'),
    Output('apply-button', 'disabled'),
    Input('apply-button', 'n_clicks'),
    Input('subject-selection', 'value'),
    Input('store-graphs', 'data'),
    State('regions-selection', 'value'),
    prevent_initial_call=True
)
def update_graph(n_clicks, name, graphs, regions):
    if (n_clicks is not None and n_clicks <= 0) or graphs is None:
        raise PreventUpdate

    return build_subject_figure(graphs[name], name, regions), True


@callback(
    Output('apply-button', 'disabled', allow_duplicate=True),
    Input('regions-selection', 'value'),
    prevent_initial_call=True
)
def enable_apply_button_at_selection_changed(regions):
    return regions is None or len(regions) == 0
