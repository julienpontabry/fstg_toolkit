from typing import Any

import numpy as np
from plotly import graph_objects as go
import dash_cytoscape as cyto

from ...graph import RC5
from ...visualization import __CoordinatesGenerator, _trans_color


def generate_subject_display_props(graph, regions: list[str]) -> dict[str, Any]:
    start_graph = graph.conditional_subgraph(t=0)

    # define nodes' properties
    nodes_coord_gen = __CoordinatesGenerator(graph)
    nodes_coord = {}
    nodes_x = []
    nodes_y = []
    nodes_color = []
    nodes_sizes = []
    levels = [0]
    all_coord = {}

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

        all_coord.update(coord)


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

    # define spatial connections
    spat_conn = {}
    for n, (x, y) in all_coord.items():
        if x not in spat_conn:
            spat_conn[x] = {}
        # NOTE use double key x/y and list to get it JSON serializable
        candidates = filter(lambda sn: graph.adj[n][sn]['type'] == 'spatial', graph.adj[n])
        candidates = filter(lambda sn: sn in all_coord, candidates)
        con_coord = [list(all_coord[sn]) for sn in candidates]
        spat_conn[x][y] = list(zip(*con_coord))  # prepare two elements for x and y coordinates

    return {
        'nodes_x': nodes_x,
        'nodes_y': nodes_y,
        'nodes_color': nodes_color,
        'nodes_sizes': nodes_sizes,
        'edges_x': edges_x,
        'edges_y': edges_y,
        'edges_colors': edges_colors,
        'levels': levels,
        'height': levels[-1] - 2,
        'regions': regions,
        'spatial_connections': spat_conn,
    }


def build_subject_figure(props: dict[str, Any]) -> go.Figure:
    # TODO find a way to reduce the number of elements displayed in the figure
    # TODO any way to improve the graphical performances (eg, WebGL)?
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
            hoverlabel=dict(bgcolor='white'),
            xaxis=dict(showgrid=False, zeroline=False, title="Time"),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=True,
                       tickvals=props['levels'][:-1], ticktext=props['regions'],
                       minor=dict(tickvals=np.subtract(props['levels'][1:-1], 1),
                                  showgrid=True, gridwidth=2, griddash='dash',
                                  gridcolor='lightgray'),
                       range=[-1.5, props['height']+1.5])
        )
    )


def build_cyto_figure(props: dict[str, Any]) -> cyto.Cytoscape:
    nodes = [
        {'data': {'id': f'{x}-{y}', 'size': 6 * size**5, 'color': color},
         'position': {'x': x * 100, 'y': y * 100},
         'selectable': False,
         'locked': True}
        for x, y, size, color in zip(props['nodes_x'], props['nodes_y'],
                                     props['nodes_sizes'], props['nodes_color'])
    ]

    edges = [
        {'data': {'source': f'{x1}-{y1}', 'target': f'{x2}-{y2}', 'color': color}}
        for (x1, x2), (y1, y2), color in zip(props['edges_x'], props['edges_y'], props['edges_colors'])
    ]

    return cyto.Cytoscape(
        layout={"name": "preset", "fit": True},
        style={"width": "100%", "height": f"{21*(props['height']+2)}px"},
        elements=nodes + edges,
        stylesheet=[
            {
                'selector': 'node',
                'style': {
                    'background-color': 'mapData(color, -1, 1, blue, red)',
                    'line-color': 'mapData(color, -1, 1, blue, red)',
                    'width': 'mapData(size, 0, 6, 1, 50)',
                    'height': 'mapData(size, 0, 6, 1, 50)',
                },
            },
            {
                'selector': 'edge',
                'style': {
                    'background-color': 'data(color)',
                    'line-color': 'data(color)',
                }
            }
        ],
        # responsive=True,
        zoom=0.1
    )
