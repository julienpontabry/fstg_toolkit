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


def build_cyto_figure(props: dict[str, Any]) -> cyto.Cytoscape:
    regions = [
        # {'data': {'id': 'insula', 'label': 'Insula'}, 'classes': 'regions'},
        # {'data': {'id': 'midbrain', 'label': 'Midbrain'}, 'classes': 'regions'}

        # {'data': {'id': region}, 'classes': 'regions'}
        # for region in props['regions']
    ]

    nodes = [
        {'data': {'id': f'{x}-{y}', 'size': 6 * size**5, 'color': color, 'parent': 'insula' if y < 10 else 'midbrain'},
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
        id='subject-cyto',
        layout={'name': 'preset', 'fit': True},
        style={'width': '100%', 'height': f'{21*(props['height']+2)}px'},
        elements=regions + nodes + edges,
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
                'selector': '.regions',
                'style': {
                    'content': 'data(label)'
                }
            },
            {
                'selector': 'edge',
                'style': {
                    'background-color': 'data(color)',
                    'line-color': 'data(color)',
                }
            }
        ],
        zoom=0.1
    )

# TODO test grouping the nodes per regions => slowdown / impractical to design (CSS)
# TODO test user mouse interactions => id not found but still capture event / not easy to update color without updating everything
# TODO global drawing (time axis, viewport, etc.) => not possible natively
