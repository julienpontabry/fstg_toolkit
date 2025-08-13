from typing import Any

import numpy as np
import pandas as pd
from plotly import graph_objects as go

from ...graph import RC5
from ...visualization import __CoordinatesGenerator, _trans_color
from ..core.color import HueInterpolator
from ..core.geometry import Arc, ArcShape, Line, LineShape


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
        con_coord = map(lambda sn: [all_coord[sn][1], graph[n][sn]['correlation']], candidates)
        spat_conn[x][y] = list(zip(*con_coord))  # prepare two elements for y coordinates and edge correlation/weight

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
                    colorscale='RdBu_r', showscale=True,
                    colorbar=dict(
                        title=dict(text="Internal strength", side='right')
                    ))
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

    # ticks for region labels
    multi_lines_regions = [r.replace(' ', '<br>') for r in props['regions']]
    centered_ticks = [(l1+l2)/2-1 for l1, l2 in zip(props['levels'][:-1], props['levels'][1:])]

    # colors bars to encompass region elements
    regions_lengths = np.diff(props['levels']).astype(float)
    region_lines = Line.from_proportions(regions_lengths/regions_lengths.sum(), total_length=props['levels'][-1],
                                         orientation=np.pi/2, origin=(-20, -1))
    region_line_paths = [LineShape(line, 5).to_path() for line in region_lines]
    shapes = [__create_path_props(path.to_svg(), 'gray', f'rgb{color}')
              for path, color in zip(region_line_paths, HueInterpolator().sample(len(region_line_paths)))]

    return go.Figure(
        data=[*edges_traces, nodes_trace],
        layout=go.Layout(
            plot_bgcolor='white',
            height=21*(props['height']+2)+126,
            margin=dict(t=40),
            showlegend=False,
            hovermode='closest',
            hoverlabel=dict(bgcolor='white'),
            xaxis=dict(showgrid=False, zeroline=False, title="Time"),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=True,
                       tickvals=centered_ticks, ticktext=multi_lines_regions,
                       range=[-1.5, props['height']+1.5]),
            shapes=shapes
        )
    )


def __create_path_props(path: str, line_color: str, fill_color: str):
    return dict(
        line=dict(color=line_color, width=0.45),
        path=path,
        type='path',
        fillcolor=fill_color,
        layer='below')


def build_spatial_figure(areas: pd.DataFrame, selected_regions: list[str], gap_size: float = 0.005,
                         fig_size: int = 500, regions_thickness: float = 0.1) -> go.Figure:
    # get regions and their areas count
    areas_sorted = areas[areas['Name_Region'].isin(selected_regions)].sort_values(by='Name_Region')
    regions = areas_sorted.groupby(by='Name_Region').count()

    # create region arcs
    proportions = regions['Name_Area'] / regions['Name_Area'].sum()
    arcs = Arc.from_proportions(proportions.to_list(), gap_size)

    # create the displayed elements for region arcs
    colors = HueInterpolator().sample(len(arcs))
    labels = regions.index.to_list()
    arcs_lines = []
    shapes = []

    for label, arc, color in zip(labels, arcs, colors):
        # create a shape from the arc
        arc_shape = ArcShape(arc, regions_thickness, 1.0)

        # add the arc line to the scatter plot
        arc_line = go.Scatter(x=arc_shape.exterior_edge.real,
                              y=arc_shape.exterior_edge.imag,
                              mode='lines',
                              line=dict(color=f'rgb{color}', shape='spline', width=0.25),
                              text=label,
                              hoverinfo='text')
        arcs_lines.append(arc_line)

        # add the shape of the path to the plate
        path = arc_shape.to_path()
        shapes.append(__create_path_props(path.to_svg(), 'gray', f'rgb{color}'))

    # build the figure object
    axis = dict(showline=False, zeroline=False, showgrid=False, showticklabels=False, title="")

    return go.Figure(
        data=arcs_lines,
        layout=go.Layout(
            plot_bgcolor='white',
            xaxis=dict(axis),
            yaxis=dict(axis),
            showlegend=False,
            width=fig_size,
            height=fig_size,
            margin=dict(t=25, b=25, l=25, r=25),
            hovermode='closest',
            shapes=shapes))
