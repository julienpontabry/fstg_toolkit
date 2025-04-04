import io

import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, dcc, html
from fmri_st_graph import spatio_temporal_graph_from_corr_matrices
from fmri_st_graph.graph import RC5
from fmri_st_graph.visualization import __CoordinatesGenerator, _trans_color
from plotly import graph_objects as go


def generate_subject_display_props(graph, name: str, regions: list[str]) -> dict[str, any]:
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


layout = html.Div([
    dcc.Dropdown([], clearable=False, id='subject-selection'),
    dcc.Dropdown([], multi=True, placeholder="Select regions...", id='regions-selection'),
    dcc.Graph(figure={}, id='st-graph')
])


@callback(
    Output('regions-selection', 'options'),
    Output('regions-selection', 'value'),
    Input('store-desc', 'data'),
)
def update_regions(desc_json):
    if desc_json is None:
        return [], []

    desc = pd.read_json(io.StringIO(desc_json))
    regions = desc.sort_values("Name_Region")["Name_Region"].unique().tolist()
    return regions, regions


@callback(
Output('subject-selection', 'options'),
    Output('subject-selection', 'value'),
    Input('store-corr', 'data'),
)
def update_subjects(corr):
    if corr is None or len(corr) == 0:
        return [], []

    return list(corr.keys()), next(iter(corr.keys()))


@callback(
    Output('st-graph', 'figure'),
    Input('subject-selection', 'value'),
    Input('regions-selection', 'value'),
    State('store-desc', 'data'),
    State('store-corr', 'data'),
)
def update_graph(name, regions, desc_json, corr):
    if desc_json is None or corr is None:
        return {}

    # FIXME very slow: we should keep the data on the server to avoid sending back and forth
    # for instance use ServerSideOutputTransform from dash extensions
    desc = pd.read_json(io.StringIO(desc_json)) # TODO put all reading in a utility function
    matrices = np.array(corr[name])
    graph = spatio_temporal_graph_from_corr_matrices(matrices, desc)
    return build_subject_figure(graph, name, regions)
