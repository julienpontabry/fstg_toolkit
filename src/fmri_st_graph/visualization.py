"""Plotting the spatio-temporal graphs."""
from cmath import isclose
from functools import cache

import networkx as nx
import numpy as np
from matplotlib import colormaps as cm
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch

from .graph import SpatioTemporalGraph, RC5


# FIXME do we need a dist parameter?
def __time_multipartite_layout(g: SpatioTemporalGraph, dist=1.0):
    pos = {}

    for t in range(g.graph['min_time'],
                   g.graph['max_time'] + 1):
        sub_g = g.conditional_subgraph(t=t)
        nodes = sorted(sub_g.nodes)
        half_height = dist * (len(nodes) - 1) / 2
        heights = np.linspace(-half_height, half_height, len(nodes))

        for n, height in zip(nodes, heights):
            pos[n] = (t, height)

    return pos


# TODO add time scale at the bottom
def multipartite_plot(g: SpatioTemporalGraph):
    plt.figure()
    pos = __time_multipartite_layout(g)
    node_color = [d['internal_strength']
                  for _, d in g.nodes.items()]
    edge_color = ['red' if d['type'] == 'temporal' else 'limegreen'
                  for _, d in g.edges.items()]
    edge_labels = {e: d['correlation'] if d['type'] == 'spatial' else d['transition']
                   for e, d in g.edges.items()}
    edge_widths = [np.abs(d['correlation']) * 4 if d['type'] == 'spatial' else 2
                   for _, d in g.edges.items()]
    connectionstyle = 'arc3'
    nx.draw_networkx(g, pos=pos, with_labels=True, node_color=node_color,
                     cmap='coolwarm', vmin=-1, vmax=1, edge_color=edge_color,
                     connectionstyle=connectionstyle, width=edge_widths)
    nx.draw_networkx_edge_labels(g, pos=pos, edge_labels=edge_labels,
                                 connectionstyle=connectionstyle)


def __polar2cart(angles: np.array, distance: float) -> tuple[np.array, np.array]:
    """Calculate the cartesian coordinates from polar ones.

    Parameters
    ----------
    angles: np.array
        The angles in radian.
    distance: float
        The distance in matplotlib's unit.

    Returns
    -------
    tuple[np.array, np.array]
        The cartesian coordinates.
    """
    pts = distance * np.exp(1j * angles)
    return np.real(pts), np.imag(pts)


def __readable_angled_annotation(angle: float) -> dict[str, float | str]:
    """Get annotation's properties depending on the display angle.

    Parameters
    ----------
    angle: float
        The angle in degrees.

    Returns
    -------
    dict[str, float | str]
        The properties for areas annotation.
    """
    if angle <= 90 or angle >= 270:
        return dict(rotation=angle, ha='left')
    else:
        return dict(rotation=angle+180, ha='right')


def __edge_con_style(angle1: float, angle2: float, bending: float = 5) -> str:
    """Get the connection style property for edges.

    Parameters
    ----------
    angle1: float
        The angle of the first node in radians.
    angle2: float
        The angle of the second node in radians.
    bending: float
        The bending coefficient (close to 0 means fully bend and close to
        infinity means straight).

    Returns
    -------
    str
        The appropriate connection style property.
    """
    diff = angle1 - angle2
    if diff > np.pi or diff < -np.pi:
        sign = -np.sign(diff)
        dist = (2*np.pi - abs(diff)) / np.pi
    else:
        sign = np.sign(diff)
        dist = abs(diff) / np.pi
    return f'arc3, rad={sign * (1 - dist) ** bending}'


def __annot_con_style(angle1: float, angle2: float) -> str:
    """Get the connection style property for annotations.

    Parameters
    ----------
    angle1: float
        The angle of the line on the side of the annotation text in degrees.
    angle2: float
        The angle of the line on the network side in degrees.

    Returns
    -------
    str
        The appropriate connection style property.
    """
    abs_diff = abs(angle1 - angle2)
    diff = min(abs_diff, 360 - abs_diff)
    if isclose(angle1, angle2) or isclose(diff, 180) or diff > 180:
        return 'arc3'
    else:
        return f'angle, angleA={angle1}, angleB={angle2}, rad=0'


def __angle_between(vec1: tuple[float, float], vec2: tuple[float, float]) -> float:
    """Calculate the angle between two vectors.

    Parameters
    ----------
    vec1: tuple[float, float]
        The first vector.
    vec2: tuple[float, float]
        The second vector.

    Returns
    -------
    float
        The angle in degrees.
    """
    angle = np.arctan2(vec2[1] - vec1[1], vec2[0] - vec1[0])
    if angle < 0:
        angle += 2 * np.pi
    return np.rad2deg(angle)


def spatial_plot(graph: SpatioTemporalGraph, t: float, ax: Axes = None, edges_bending: float = 5) -> None:
    """Draw a spatial plot for spatio-temporal graph.

    Parameters
    ----------
    graph: SpatioTemporalGraph
        The graph to plot.
    t: float
        The instant to plot in the graph.
    ax: matplotlib.axes.Axes, optional
        The axes on which to plot.
    edges_bending: float, optional
        Controls the bending of the edges. Close to 0 means full bending, close to
        infinity means no bending (default is 5).
    """
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
    ax.pie([len(rels[rels['Name_Region'] == region]) / n
            for region in regions],
           labels=regions, radius=2.25, startangle=-360 / n / 2,
           colors=[regions_cmap(i) for i in range(len(regions))],
           wedgeprops=dict(width=1, edgecolor='w', alpha=0.3))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    # plot networks
    cmap = cm.get_cmap('coolwarm')
    nodes_angles = {}
    nodes_angles_map = {}
    nodes_coords = {}
    areas_network_map = {}
    for node, data in sub_g.nodes.items():
        network = list(data['areas'])
        indices = np.argwhere(np.isin(rels.index, network)).flatten().tolist()

        angle = angles[indices].mean()
        closest_node = min([(on, abs(a-angle)) for on, a in nodes_angles.items()], default=(None, -1), key=lambda e: e[1])
        if isclose(closest_node[1], 0):
            angle += 2*np.pi/n/2
        nodes_angles[node] = angle
        nodes_angles_map |= {i: angle for i in indices}

        x, y = __polar2cart(angle, 1)
        corr = data['internal_strength']
        ax.plot(x, y, 'o', mfc=cmap(corr / 2 + 0.5),
                mec='k', ms=(15 * np.abs(corr)), zorder=4)
        nodes_coords[node] = (x, y)
        areas_network_map |= {i: (x, y) for i in indices}

    # plot areas' labels
    for i, (x_area, y_area, area) in enumerate(zip(x_areas, y_areas, rels['Name_Area'])):
        to_node_angle = __angle_between(areas_network_map[i], __polar2cart(angles[i], 1.2))
        angle = np.rad2deg(angles[i])
        ax.annotate(text=area, xy=areas_network_map[i], xytext=(x_area, y_area),
                    va='center', fontsize='x-small', rotation_mode='anchor',
                    **__readable_angled_annotation(angle),
                    arrowprops=dict(arrowstyle='-',
                                    connectionstyle=__annot_con_style(angle, to_node_angle),
                                    linestyle=':'))

    # plot edges between networks
    for (n1, n2), d in sub_g.edges.items():
        edge_patch = FancyArrowPatch(
            posA=nodes_coords[n1], posB=nodes_coords[n2],arrowstyle='-',
            connectionstyle=__edge_con_style(nodes_angles[n1], nodes_angles[n2], bending=edges_bending),
            linewidth=np.abs(d['correlation'])*4, color=cmap(d['correlation']/2+0.5), alpha=np.abs(d['correlation']))
        ax.add_artist(edge_patch)


class __CoordinatesGenerator:
    def __init__(self, graph: SpatioTemporalGraph) -> None:
        self.g = graph
        self.__max_heights = None
        self.__coords = None

    @cache
    def __next_temp_trans(self, node: int) -> list[int]:
        return [m for m in self.g[node]
                if self.g[node][m]['type'] == 'temporal']

    @cache
    def __time_from_node(self, node: int) -> int:
        return self.g.nodes[node]['t']

    def __find_height_for_path(self, node: int, y: int) -> int:
        current_max_y = y
        trans = self.__next_temp_trans(node)

        while trans != [] and (m := trans[0]) not in self.__coords:
            t = self.__time_from_node(m)

            if t in self.__max_heights:
                current_max_y = max(current_max_y, self.__max_heights[t] + 1)

            trans = self.__next_temp_trans(m)

        return current_max_y

    def __generate_coords_for_node(self, node: int, base_y: int) -> tuple[int, int]:
        return self.__time_from_node(node), self.__find_height_for_path(node, base_y)

    def __generate_coords_for_path_rec(self, trans_list: list[int], base: int) -> None:
        base_y = base

        for i, m in enumerate(trans_list):
            if m not in self.__coords:
                t, y = self.__generate_coords_for_node(m, base_y)
                base_y += 1
                self.__coords[m] = (t, y)
                self.__max_heights[t] = y
                next_trans = self.__next_temp_trans(m)
                self.__generate_coords_for_path_rec(next_trans, y)

    def generate(self, nodes: list[int], base_y: int) -> dict[int, tuple[int, int]]:
        self.__max_heights = dict()
        self.__coords = {n: (self.g.nodes[n]['t'], base_y + i) for i, n in enumerate(nodes)}

        for i, node in enumerate(nodes):
            trans_list = self.__next_temp_trans(node)
            self.__generate_coords_for_path_rec(trans_list, base_y + i)

        return self.__coords


@cache
def _trans_color(transition: RC5) -> str:
    if transition == RC5.PP:
        return 'red'
    elif transition == RC5.PPi:
        return 'blue'
    elif transition == RC5.PO:
        return 'limegreen'
    else:
        return 'black'


class __PathDrawer:
    def __init__(self, g: SpatioTemporalGraph, axe: Axes) -> None:
        self.g = g
        self.axe = axe
        self.__done = None
        self.__lines = None
        self.__colors = None

    @cache
    def __next_temp_trans(self, node: int) -> list[tuple[int, RC5]]:
        return [(m, self.g[node][m]['transition'])
                for m in self.g[node]
                if self.g[node][m]['type'] == 'temporal']

    def __draw_rec(self, coords: dict[int, tuple[int, int]], trans_list: list[tuple[int, RC5]],
                   node: int, prev_t: int, prev_y: int) -> None:
        for m, rc5 in trans_list:
            if m in coords and (node, m) not in self.__done:
                t, y = coords[m]
                self.__lines.append([(prev_t, prev_y), (t, y)])
                self.__colors.append(_trans_color(rc5))
                self.__done.add((node, m))
                next_trans = self.__next_temp_trans(m)
                self.__draw_rec(coords, next_trans, m, t, y)

    def draw(self, coords: dict[int, tuple[int, int]], nodes: list[int], base_y: int) -> None:
        self.__done = set()
        self.__lines = list()
        self.__colors = list()

        for i, node in enumerate(nodes):
            trans_list = self.__next_temp_trans(node)
            self.__draw_rec(coords, trans_list, node, self.g.nodes[node]['t'], base_y + i)

        self.axe.add_collection(
            LineCollection(self.__lines, colors=self.__colors,
                           linewidths=1.5, linestyles='-'))


# TODO add time scale at the bottom
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
    gen = __CoordinatesGenerator(graph)
    drawer = __PathDrawer(graph, ax)
    for r, region in enumerate(regions):
        nodes = [n for n, d in sub_g.nodes.items() if d['region'] == region]
        coords = gen.generate(nodes, y)
        drawer.draw(coords, nodes, y)

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
