from pathlib import Path

import numpy as np
import pandas as pd
from dash import dcc, html
from fmri_st_graph import spatio_temporal_graph_from_corr_matrices
from fmri_st_graph.graph import RC5
from fmri_st_graph.visualization import __CoordinatesGenerator, _trans_color
from plotly import graph_objects as go


class LifeLinesGenerator:
    def generate(self, starting_nodes: list[int], y: int):
        pass


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
        nodes_x += x  # FIXME keep the structure of life lines
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

    nodes_trace = go.Scatter(
        x=props['nodes_x'], y=props['nodes_y'],
        mode='markers',
        hoverinfo='none',
        marker=dict(size=6*np.power(props['nodes_sizes'], 5),
                    color=props['nodes_color'],
                    cmin=-1, cmid=0, cmax=1, line_width=0,
                    colorscale='RdBu_r', showscale=True)
    )

    # FIXME reduce the number of traces (7831!)
    # drop these and use nodes traces per life line
    # colors may be dropped
    # => currently not displaying all the edges, only the non-EQ ones
    edges_traces = []
    for x, y, c in zip(props['edges_x'], props['edges_y'], props['edges_colors']):
        edges_trace = go.Scatter(
            x=x, y=y,
            mode='lines',
            hoverinfo='skip',
            line=dict(width=0.5, color=c)
        )
        edges_traces.append(edges_trace)
    print(len(edges_traces))
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


data_path = Path('/home/jpontabry/Documents/projets/visualisation graphes spatio-temporels/data')
desc = pd.read_csv(data_path / 'brain_areas_regions_rel_full.csv', index_col=0)
data = np.load(data_path / 'list_of_corr_matrices_5months.zip')
matrices = next(iter(data.values()))
graph = spatio_temporal_graph_from_corr_matrices(matrices, desc)

rels = desc.sort_values("Name_Region")
regions = rels["Name_Region"].unique().tolist()
# regions = ['Thalamus', 'Midbrain', 'Retrosplenial']

props = generate_subject_display_props(graph, 'test', regions)
figure = build_subject_figure(graph, 'test', regions)

layout = html.Div([
    dcc.Graph(figure=figure, id='stgraph')
])
