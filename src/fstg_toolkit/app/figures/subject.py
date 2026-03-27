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

from collections import defaultdict
from math import floor, ceil
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import cm as _mpl_cm
from plotly import graph_objects as go

from .common import norm_min_max_size
from ..core.color import HueInterpolator
from ..core.geometry import Arc, ArcShape, Line, LineShape, Ribbon, RibbonShape
from ... import SpatioTemporalGraph
from ...graph import RC5
from ...visualization import __CoordinatesGenerator, _trans_color

NODE_PROP_LABELS: dict[str, str] = {
    'internal_strength': 'Internal strength',
    'efficiency': 'Efficiency',
}


def generate_temporal_graph_props(graph: SpatioTemporalGraph, regions: list[str],
                                  color_prop: str = 'internal_strength',
                                  size_prop: str = 'efficiency') -> dict[str, Any]:
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
        nodes_color += [graph.nodes[n][color_prop] for n in coord.keys()]
        nodes_sizes += [graph.nodes[n][size_prop] for n in coord.keys()]
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
        candidates = filter(lambda sn, n=n: graph.adj[n][sn]['type'] == 'spatial', graph.adj[n])
        candidates = filter(lambda sn: sn in all_coord, candidates)
        con_coord = [[all_coord[sn][1], graph[n][sn]['correlation']] for sn in candidates]
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
        'color_label': NODE_PROP_LABELS.get(color_prop, color_prop),
    }


def build_subject_figure(props: dict[str, Any], areas: pd.Series) -> go.Figure:
    def __scale_size(s: list[float], power: float = 5, max_size: float = 6) -> list[float]:
        s = norm_min_max_size(s)
        return (max_size * np.power(s, power)).tolist()

    nodes_trace = go.Scatter(
        x=props['nodes_x'], y=props['nodes_y'],
        mode='markers',
        hoverinfo='text',
        hovertext=["<br>".join([areas.loc[i] for i in s])
                   for s in props['nodes_areas']],
        marker={'size': __scale_size(props['nodes_sizes']),
                'color': props['nodes_color'],
                'cmin': floor(min(props['nodes_color'])), 'cmax': ceil(max(props['nodes_color'])), 'line_width': 0,
                'colorscale': 'RdBu_r', 'showscale': True,
                'colorbar': {
                    'title': {'text': props.get('color_label', 'Internal strength'), 'side': 'right'}
                }}
    )

    edges_traces = []
    for x, y, c in zip(props['edges_x'], props['edges_y'], props['edges_colors']):
        edges_trace = go.Scatter(
            x=x, y=y,
            mode='lines',
            hoverinfo='skip',
            line={'width': 0.5, 'color': c}
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
            margin={'t': 40},
            showlegend=False,
            hovermode='closest',
            hoverlabel={'bgcolor': 'white'},
            xaxis={'showgrid': False, 'zeroline': False, 'title': "Time"},
            yaxis={'showgrid': False, 'zeroline': False, 'showticklabels': True,
                   'tickvals': centered_ticks, 'ticktext': multi_lines_regions,
                   'range': [-1.5, props['height']+1.5]},
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
    nodes_areas_count = ([len(s) for s in l] for l in nodes_areas)
    nodes_proportions = [[c/sum(l) for c in l] for l in nodes_areas_count]
    nodes_areas_labels = [[[areas_desc["Name_Area"].loc[n] for n in s] for s in l] for l in nodes_areas]

    # build mapping from node ID to (region_idx, node_idx_within_region) for ribbons
    node_to_arc_idx = {}
    for region_idx, region_name in enumerate(regions.index):
        for node_idx, (node_id, _) in enumerate(graph.sub(region=region_name).nodes.items()):
            node_to_arc_idx[node_id] = (region_idx, node_idx)

    # create ribbon specifications from spatial edges (one per undirected pair)
    seen_pairs: set[tuple[int, int]] = set()
    ribbon_specs = []
    for n, m, d in graph.edges(data=True):
        if d['type'] == 'spatial' and n in node_to_arc_idx and m in node_to_arc_idx:
            pair = (min(n, m), max(n, m))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                ribbon_specs.append({
                    'source': list(node_to_arc_idx[n]),
                    'target': list(node_to_arc_idx[m]),
                    'correlation': d['correlation']
                })

    return {
        'region_labels': regions.index.to_list(),
        'region_proportion': region_proportions.tolist(),
        'nodes_labels': nodes_areas_labels,
        'nodes_proportions': nodes_proportions,
        'ribbons': ribbon_specs
    }


def __corr_to_rgba(correlation: float) -> str:
    cmap = _mpl_cm.get_cmap('RdBu_r')
    r, g, b, _ = cmap((correlation + 1) / 2)
    alpha = abs(correlation) * 0.5 + 0.1
    return f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {alpha:.3f})'


def __create_path_props(path: str, line_color: str, fill_color: str) -> dict[str, Any]:
    return {
        'line': {'color': line_color, 'width': 0.45},
        'path': path,
        'type': 'path',
        'fillcolor': fill_color,
        'layer': 'below'
    }


def __create_arc_elements(arc: Arc, thickness: float, radius: float, label: str, fill_color: str, line_color: str = 'gray') -> tuple[go.Scatter, dict[str, Any]]:
    # create a shape from the arc
    arc_shape = ArcShape(arc, thickness, radius)

    # create scatter plot from the arc shape
    arc_line = go.Scatter(x=arc_shape.exterior_edge.real,
                          y=arc_shape.exterior_edge.imag,
                          mode='lines',
                          line={'color': fill_color, 'shape': 'spline', 'width': 0.25},
                          text=label,
                          hoverinfo='text')

    # add the shape of the path to the plate
    path = arc_shape.to_path()
    path_props = __create_path_props(path.to_svg(), line_color, fill_color)

    return arc_line, path_props


def __create_ribbon_elements(nodes_arcs: list[list[Arc]], radius: float, ribbons, thickness: float) -> list[Any]:
    node_ribbon_list: dict[tuple, list] = defaultdict(list)  # (ri, ni) -> [(ribbon_idx, weight)]
    for idx, spec in enumerate(ribbons):
        w = max(abs(spec['correlation']), 0.01)
        node_ribbon_list[tuple(spec['source'])].append((idx, w))
        node_ribbon_list[tuple(spec['target'])].append((idx, w))

    ribbon_sub_arcs: dict[int, dict[str, tuple[float, float]]] = {}
    for node_key, rib_list in node_ribbon_list.items():
        ri, ni = node_key
        arc = nodes_arcs[ri][ni]
        total_w = sum(w for _, w in rib_list)
        current = arc.begin

        current_midpoint = (arc.begin + arc.end) / 2

        def __other_end_angle(item: tuple[int, float],
                              nk: tuple[Any] = node_key,
                              orig_mid_point: float = current_midpoint) -> float:
            sp = ribbons[item[0]]
            other = sp['target'] if tuple(sp['source']) == nk else sp['source']
            other_arc = nodes_arcs[other[0]][other[1]]
            other_midpoint = (other_arc.begin + other_arc.end) / 2
            return (other_midpoint - orig_mid_point) % (2 * np.pi)

        for idx, w in sorted(rib_list, key=__other_end_angle, reverse=True):
            sub_end = current + (w / total_w) * arc.angle
            side = 'src' if tuple(ribbons[idx]['source']) == node_key else 'tgt'
            ribbon_sub_arcs.setdefault(idx, {})[side] = (current, sub_end)
            current = sub_end

    # create ribbon traces from spatial edges (rendered below arcs)
    ribbon_traces = []
    for idx, spec in enumerate(ribbons):
        angles = ribbon_sub_arcs.get(idx, {})
        src_arc = nodes_arcs[spec['source'][0]][spec['source'][1]]
        tgt_arc = nodes_arcs[spec['target'][0]][spec['target'][1]]
        src_b, src_e = angles.get('src', (src_arc.begin, src_arc.end))
        tgt_b, tgt_e = angles.get('tgt', (tgt_arc.begin, tgt_arc.end))
        ribbon_shape = RibbonShape(Ribbon(src_b, tgt_e),
                                   Ribbon(src_e, tgt_b),
                                   radius=radius - thickness, strength=0.3)
        path = ribbon_shape.to_path()
        if path.points:
            xs, ys = zip(*path.points)
            ribbon_traces.append(go.Scatter(
                x=list(xs), y=list(ys),
                fill='toself',
                fillcolor=__corr_to_rgba(spec['correlation']),
                line={'color': 'rgba(0,0,0,0)', 'width': 0},
                mode='lines',
                hoverinfo='text',
                text=f"Correlation: {spec['correlation']:.2f}",
                showlegend=False
            ))
    return ribbon_traces


def build_spatial_figure(props: dict[str, Any], gap_size: float = 0.005,
                         thickness: float = 0.1, radius: float = 1.0) -> go.Figure:
    # create region arcs
    region_arcs = Arc.from_proportions(props['region_proportion'], gap_size)
    nodes_arcs = [Arc.from_proportions(region_props, begin=arc.begin, length=arc.angle)
                  for arc, region_props in zip(region_arcs, props['nodes_proportions'])]

    # compute proportional sub-arc allocations for each ribbon endpoint
    ribbons = props.get('ribbons', [])
    ribbon_traces = __create_ribbon_elements(nodes_arcs, radius, ribbons, thickness)

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
    axis = {'showline': False, 'zeroline': False, 'showgrid': False, 'showticklabels': False, 'title': ""}

    return go.Figure(
        data=[*ribbon_traces, *arcs_lines],
        layout=go.Layout(
            plot_bgcolor='white',
            autosize=True,
            xaxis=dict(axis, scaleanchor='y', scaleratio=1),
            yaxis=dict(axis),
            showlegend=False,
            margin={'t': 25, 'b': 25, 'l': 25, 'r': 25},
            hovermode='closest',
            shapes=shapes))
