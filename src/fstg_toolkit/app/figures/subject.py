# Copyright 2025 ICube (University of Strasbourg - CNRS)
# author: Julien PONTABRY (ICube)
#
# This software is a computer program whose purpose is to provide a toolkit
# to model, process and analyze the longitudinal reorganization of brain
# connectivity data, as functional MRI for instance.
#
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/or redistribute the software under the terms of the CeCILL-B
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL-B license and that you accept its terms.

from typing import Any

import numpy as np
import pandas as pd
from plotly import graph_objects as go

from ... import SpatioTemporalGraph
from ...graph import RC5
from ...visualization import __CoordinatesGenerator, _trans_color
from ..core.color import HueInterpolator
from ..core.geometry import Arc, ArcShape, Line, LineShape


def generate_temporal_graph_props(graph: SpatioTemporalGraph, regions: list[str]) -> dict[str, Any]:
    start_graph = graph.sub(t=0)

    # define nodes' properties
    nodes_coord_gen = __CoordinatesGenerator(graph)
    nodes_coord = {}
    nodes_x = []
    nodes_y = []
    nodes_color = []
    nodes_sizes = []
    nodes_areas = []
    levels = [0]
    all_coord = {}

    for region in regions:
        nodes = [n for n, d in start_graph.nodes.items() if d['region'] == region]
        coord = nodes_coord_gen.generate(nodes, levels[-1])
        x, y = tuple(zip(*coord.values()))
        nodes_color += [graph.nodes[n]['internal_strength'] for n in coord.keys()]
        nodes_sizes += [graph.nodes[n]['efficiency'] for n in coord.keys()]
        nodes_areas += [graph.nodes[n]['areas'] for n in coord.keys()]

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
        'nodes_areas': nodes_areas,
        'edges_x': edges_x,
        'edges_y': edges_y,
        'edges_colors': edges_colors,
        'levels': levels,
        'height': levels[-1] - 2,
        'regions': regions,
        'spatial_connections': spat_conn,
    }


def build_subject_figure(props: dict[str, Any], areas: pd.Series) -> go.Figure:
    nodes_trace = go.Scatter(
        x=props['nodes_x'], y=props['nodes_y'],
        mode='markers',
        hoverinfo='text',
        hovertext=["<br>".join(map(lambda i: areas.loc[i], s))
                   for s in props['nodes_areas']],
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


def generate_spatial_graph_props(graph: SpatioTemporalGraph, areas_desc: pd.DataFrame, regions: list[str]) -> dict[str, Any]:
    # get regions and their areas count
    areas_sorted = areas_desc[areas_desc["Name_Region"].isin(regions)].sort_values(by="Name_Region")
    regions = areas_sorted.groupby(by="Name_Region").count()

    # calculate arc proportions for regions
    region_proportions = regions["Name_Area"] / regions["Name_Area"].sum()

    # calculate arc proportions for all nodes within regions
    nodes_areas = [[d['areas']
                    for _, d in graph.sub(region=region).nodes.items()]
                   for region in regions.index]
    nodes_areas_count = map(lambda l: [len(s) for s in l], nodes_areas)
    nodes_proportions = map(lambda l: [c/sum(l) for c in l], nodes_areas_count)
    nodes_areas_labels = map(lambda l: [[areas_desc["Name_Area"].loc[n] for n in s] for s in l], nodes_areas)

    # TODO create ribbons properties for spatial edges

    return {
        'region_labels': regions.index.to_list(),
        'region_proportion': region_proportions.tolist(),
        'nodes_labels': list(nodes_areas_labels),
        'nodes_proportions': list(nodes_proportions)
    }


def __create_path_props(path: str, line_color: str, fill_color: str) -> dict[str, Any]:
    return dict(
        line=dict(color=line_color, width=0.45),
        path=path,
        type='path',
        fillcolor=fill_color,
        layer='below')


def __create_arc_elements(arc: Arc, thickness: float, radius: float, label: str, fill_color: str, line_color: str = 'gray') -> tuple[go.Scatter, dict[str, Any]]:
    # create a shape from the arc
    arc_shape = ArcShape(arc, thickness, radius)

    # create scatter plot from the arc shape
    arc_line = go.Scatter(x=arc_shape.exterior_edge.real,
                          y=arc_shape.exterior_edge.imag,
                          mode='lines',
                          line=dict(color=fill_color, shape='spline', width=0.25),
                          text=label,
                          hoverinfo='text')

    # add the shape of the path to the plate
    path = arc_shape.to_path()
    path_props = __create_path_props(path.to_svg(), line_color, fill_color)

    return arc_line, path_props


def build_spatial_figure(props: dict[str, Any], gap_size: float = 0.005,
                         fig_size: int = 500, thickness: float = 0.1, radius: float = 1.0) -> go.Figure:
    # create region arcs
    region_arcs = Arc.from_proportions(props['region_proportion'], gap_size)
    nodes_arcs = [Arc.from_proportions(region_props, begin=arc.begin, length=arc.angle)
                  for arc, region_props in zip(region_arcs, props['nodes_proportions'])]

    # create the displayed elements for arcs
    colors = HueInterpolator().sample(len(region_arcs))
    arcs_lines = []
    shapes = []

    for reg_label, arc, color, subarcs, subarcs_labels in zip(
            props["region_labels"], region_arcs, colors, nodes_arcs, props["nodes_labels"]):
        # display region arcs
        region_arc_line, region_arc_path = __create_arc_elements(
            arc, thickness, radius, reg_label, f'rgb{color}')
        arcs_lines.append(region_arc_line)
        shapes.append(region_arc_path)

        # display nodes arcs
        for subarc, subarc_label in zip(subarcs, subarcs_labels):
            subarc_label = "<br>".join(subarc_label)
            node_arc_line, node_arc_path = __create_arc_elements(
                subarc, thickness, radius - thickness, subarc_label, 'lightgray')
            arcs_lines.append(node_arc_line)
            shapes.append(node_arc_path)

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
