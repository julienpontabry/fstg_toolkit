"""Plotting the spatio-temporal graphs."""

import numpy as np
from matplotlib import colormaps as cm
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from .graph import SpatioTemporalGraph, RC5


def __polar2cart(angles, distance):
    pts = distance * np.exp(1j * angles)
    return np.real(pts), np.imag(pts)


def __readable_angled_annotation(angle):
    if angle <= 90 or angle >= 270:
        return dict(rotation=angle, ha='left')
    else:
        return dict(rotation=angle+180, ha='right')


def spatial_plot(graph: SpatioTemporalGraph, t: float, ax: Axes = None) -> None:
    if ax is None:
        ax = plt.gca()
    ax.axis('off')

    sub_g = graph.conditional_subgraph(t=t)

    rels = graph.areas.sort_values('Name_Region')
    regions = rels['Name_Region'].unique()

    # calculate positions for areas
    n = len(rels)
    angles = 2 * np.pi / n * np.arange(n)
    x_areas, y_areas = __polar2cart(angles, 1.5)

    # plot regions in a pie
    regions_cmap = cm.get_cmap('tab20')
    ax.pie([len(rels[rels['Name_Region'] == region]) / len(rels)
            for region in regions],
           labels=regions, radius=2.25, startangle=-360 / n / 2,
           colors=[regions_cmap(i) for i in range(len(regions))],
           wedgeprops=dict(width=1, edgecolor='w', alpha=0.3))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    # plot networks
    # FIXME network nodes may be overlapping (depends on the sorting)
    cmap = cm.get_cmap('coolwarm')
    nodes_coords = {}
    areas_network_map = {}
    for n, d in sub_g.nodes.items():
        network = list(d['areas'])
        indices = np.argwhere(np.isin(rels.index, network)).flatten().tolist()
        x, y = __polar2cart(angles[indices].mean(), 1)
        corr = d['internal_strength']
        ax.plot(x, y, 'o', mfc=cmap(corr / 2 + 0.5),
                mec='k', ms=15 * np.abs(corr), zorder=4)
        nodes_coords[n] = (x, y)
        areas_network_map |= {i: (x, y) for i in indices}

    # plot areas' labels
    for i, (x_area, y_area, area) in enumerate(zip(x_areas, y_areas,
                                                   rels['Name_Area'])):
        ax.annotate(text=area, xy=areas_network_map[i], xytext=(x_area, y_area),
                    va='center', fontsize='x-small', rotation_mode='anchor',
                    **__readable_angled_annotation(np.rad2deg(angles[i])),
                    arrowprops=dict(arrowstyle='-', linestyle='--'))

    # plot edges between networks
    # TODO make curved edges
    for (e1, e2), d in sub_g.edges.items():
        ax.plot(*tuple(zip(nodes_coords[e1], nodes_coords[e2])), '-',
                lw=np.abs(d['correlation']) * 4, alpha=np.abs(d['correlation']),
                color=cmap(d['correlation'] / 2 + 0.5))


def __node_to_plot(t, r, n, cum_max_nodes):
    return t, cum_max_nodes[r] + r + n


def __next_temp_trans(stG, n):
    return [(m, stG[n][m]['transition']) for m in stG[n]
            if stG[n][m]['type'] == 'temporal']


def _walk_through(G, n):
    next_trans = __next_temp_trans(G, n)
    print(G.nodes[n]['t'], next_trans)
    inp = input()

    while inp != 'q':
        k = min(len(next_trans), int(inp)) if inp.isdigit() else 0
        n, _ = next_trans[k]
        next_trans = __next_temp_trans(G, n)
        print(G.nodes[n]['t'], next_trans)
        inp = input()


def __trans_color(trans):
    if trans == RC5.PP:
        return 'red'
    elif trans == RC5.PPi:
        return 'blue'
    elif trans == RC5.PO:
        return 'limegreen'
    else:
        return 'black'


def __find_height_for_path(G, coords, n, y):
    current_max_y = y
    trans = __next_temp_trans(G, n)

    while trans != [] and trans[0][0] not in coords:
        m, _ = trans[0]
        t = G.nodes[m]['t']

        heights = [e[1] for e in coords.values() if e[0] == t]
        if len(heights) > 0:
            current_max_y = max(current_max_y, max(heights) + 1)

        trans = __next_temp_trans(G, m)

    return current_max_y


def __generate_coords_for_node(G, coords, n, base_y):
    return G.nodes[n]['t'], __find_height_for_path(G, coords, n, base_y)


def __generate_coords_for_path(G, nodes, base_y):
    def __generate_coords_for_path_rec(G, coords, trans_list, base):
        base_y = base

        for i, (m, _) in enumerate(trans_list):
            if m not in coords:
                t, y = __generate_coords_for_node(G, coords, m, base_y)
                base_y += 1
                coords[m] = (t, y)
                next_trans = __next_temp_trans(G, m)
                __generate_coords_for_path_rec(G, coords, next_trans, y)

    coords = {n: (G.nodes[n]['t'], base_y + i) for i, n in enumerate(nodes)}
    for i, n in enumerate(nodes):
        trans_list = __next_temp_trans(G, n)
        __generate_coords_for_path_rec(G, coords, trans_list, base_y + i)

    return coords


def __draw_paths(axe, G, coords, nodes, base_y):
    done = set()

    def __draw_paths_rec(trans_list, n, prev_t, prev_y):
        for m, rc5 in trans_list:
            if m in coords and (n, m) not in done:
                t, y = coords[m]
                axe.plot([prev_t, t], [prev_y, y],
                         '-', color=__trans_color(rc5), lw=1.5)
                # plt.show()
                # plt.pause(0.1)
                done.add((n, m))
                next_trans = __next_temp_trans(G, m)
                __draw_paths_rec(next_trans, m, t, y)

    for i, n in enumerate(nodes):
        trans_list = __next_temp_trans(G, n)
        __draw_paths_rec(trans_list, n, G.nodes[n]['t'], base_y + i)

    return coords


def temporal_plot(graph: SpatioTemporalGraph, ax: Axes = None):
    if ax is None:
        ax = plt.gca()
    ax.axis('off')

    rels = graph.areas.sort_values('Name_Region')
    regions = rels['Name_Region'].unique().tolist()

    times = np.unique([d['t'] for n, d in graph.nodes.items()])

    # draw dynamic (nodes + transitions)
    cmap = cm.get_cmap('coolwarm')
    sub_g = graph.conditional_subgraph(t=0)
    heights = []
    y = 0
    for r, region in enumerate(regions):
        nodes = [n for n, d in sub_g.nodes.items() if d['region'] == region]
        coords = __generate_coords_for_path(graph, nodes, y)

        __draw_paths(ax, graph, coords, nodes, y)

        colors = [graph.nodes[n]['internal_strength'] / 2 + 0.5
                  for n in coords.keys()]
        ax.scatter(*list(zip(*coords.values())), zorder=2.1,
                s=30, c=colors, cmap=cmap, edgecolors='k', linewidths=1)

        heights.append(max(coords.values(), key=lambda x: x[1])[1] + 1 - y)
        y += heights[-1] + 1

    # draw limits of regions
    o = 0
    regions_cmap = cm.get_cmap('tab20')
    for r, region in enumerate(regions):
        m = heights[r]
        o += m
        ax.fill_between([times.min() - 0.5, times.max() + 0.5], o - m - 1, o,
                        fc=regions_cmap(r), alpha=0.2)
        ax.annotate(text=region, xy=(-1, (o - m - 1 + o) / 2),
                    va='center', fontsize='small', ha='right')
        o += 1
