"""Plotting the spatio-temporal graphs."""
from cmath import isclose
from dataclasses import dataclass
from functools import cache
from typing import Callable

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import colormaps as cm
from matplotlib import pyplot as plt
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation
from matplotlib.widgets import Cursor

from .graph import SpatioTemporalGraph, RC5


def __time_multipartite_layout(g: SpatioTemporalGraph, dist: float = 1.0) -> dict[int, tuple[int, float]]:
    """Create coordinates for all the nodes of the spatio-temporal graph.

    The nodes are placed vertical by time point.

    Parameters
    ----------
    g: SpatioTemporalGraph
        The spatio-temporal graph.
    dist: float, optional
        The distance factor for vertical space occupied by the nodes. Default is 1.

    Returns
    -------
    dict[int, tuple[int, float]]
        A dictionary of coordinates associated with nodes.
    """
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


def multipartite_plot(g: SpatioTemporalGraph, ax: Axes = None) -> None:
    """Draw a multipartite plot for the spatio-temporal graph.

    Parameters
    ----------
    g: SpatioTemporalGraph
        The spatio-temporal graph.
    ax: matplotlib.axes.Axes, optional
        The axes on which to plot. If not set, the current axes will be used.
    """
    if ax is None:
        ax = plt.gca()

    pos = __time_multipartite_layout(g)
    node_color = [d['internal_strength']
                  for _, d in g.nodes.items()]
    edge_color = []
    edge_labels = dict()
    edge_widths = []
    for e, d in g.edges.items():
        if d['type'] == 'temporal':
            edge_color.append('red')
            edge_widths.append(2)
            edge_labels[e] = d['transition']
        else:
            edge_color.append('limegreen')
            edge_widths.append(np.abs(d['correlation']) * 4)
            edge_labels[e] = d['correlation']

    # draw the graph
    nx.draw_networkx(g, ax=ax, pos=pos, with_labels=True, node_color=node_color,
                     cmap='coolwarm', vmin=-1, vmax=1, edge_color=edge_color,
                     connectionstyle='arc3', width=edge_widths, hide_ticks=False)
    nx.draw_networkx_edge_labels(g, ax=ax, pos=pos, edge_labels=edge_labels,
                                 connectionstyle='arc3', hide_ticks=False)

    # set up the axes
    ax.spines[['left', 'top', 'right']].set_visible(False)

    min_time, max_time = g.graph['min_time'], g.graph['max_time']
    ax.get_xaxis().tick_bottom()
    ax.set_xticks(range(min_time, max_time + 1))
    ax.set_xlabel("Time")
    ax.set_xlim(min_time - 0.125, max_time + 0.125)

    ax.set_yticks([])


def __polar2cart(angles: np.array, distance: float) -> tuple[np.array, np.array]:
    """Calculate the cartesian coordinates from polar ones.

    Parameters
    ----------
    angles: np.ndarray
        The angles in radian.
    distance: float
        The distance in matplotlib's unit.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
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


def __areas_positions(graph: SpatioTemporalGraph) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate positions for areas' labels in a circular manner.

    Parameters
    ----------
    graph: SpatioTemporalGraph
        The graph to plot.

    Returns
    -------
    rels: pandas.DataFrame
        The sorted areas names.
    angles: numpy.ndarray
        The areas angles around the circle.
    x_areas: numpy.ndarray
        The cartesian coordinates of areas along the x-axis.
    y_areas: numpy.ndarray
        The cartesian coordinates of areas along the y-axis.
    """
    rels = graph.areas.sort_values('Name_Region')
    n = len(rels)
    angles = 2 * np.pi / n * np.arange(n)
    x_areas, y_areas = __polar2cart(angles, 1.5)
    return rels, angles, x_areas, y_areas


def _spatial_plot_artists(graph: SpatioTemporalGraph, t: float,
                          edges_bending: float = 3) -> tuple[list[Line2D], list[FancyArrowPatch], list[FancyArrowPatch]]:
    """Generate artists for a spatial plotting of a spatio-temporal graph.

    Parameters
    ----------
    graph: SpatioTemporalGraph
        The graph to plot.
    t: float
        The instant to plot in the graph.
    edges_bending: float, optional
        Controls the bending of the edges. Close to 0 means full bending, close to
        infinity means no bending (default is 3).

    Returns
    -------
    networks_markers: list[Line2D]
        The artists for the nodes representing the networks.
    areas_patches: list[FancyArrowPatch]
        The arrows from areas to networks nodes.
    edges_patches: list[FancyArrowPatch]
        The patches representing the edges between the networks.
    """
    sub_g = graph.conditional_subgraph(t=t)

    rels, angles, x_areas, y_areas = __areas_positions(graph)
    n = len(rels)

    # networks node
    cmap = cm.get_cmap('coolwarm')
    nodes_angles = {}
    nodes_angles_map = {}
    nodes_coords = {}
    areas_network_map = {}
    networks_markers = []
    for node, data in sub_g.nodes.items():
        network = list(data['areas'])
        indices = np.argwhere(np.isin(rels.index, network)).flatten().tolist()

        angle = angles[indices].mean()
        closest_node = min([(on, abs(a-angle)) for on, a in nodes_angles.items()],
                           default=(None, -1), key=lambda e: e[1])
        if isclose(closest_node[1], 0):
            angle += 2*np.pi/n/2
        nodes_angles[node] = angle
        nodes_angles_map |= {i: angle for i in indices}

        x, y = __polar2cart(angle, 1)
        corr = data['internal_strength']
        l = Line2D([x], [y], marker='o', mfc=cmap(corr / 2 + 0.5),
                   mec='k', ms=(15 * np.abs(corr)), zorder=4)
        networks_markers.append(l)
        nodes_coords[node] = (x, y)
        areas_network_map |= {i: (x, y) for i in indices}

    # areas' links
    areas_patches = []
    for i, (x_area, y_area) in enumerate(zip(x_areas, y_areas)):
        to_node_angle = __angle_between(areas_network_map[i], __polar2cart(angles[i], 1.2))
        angle = np.rad2deg(angles[i])
        a = FancyArrowPatch(posA=(x_area, y_area), posB=areas_network_map[i],
                            arrowstyle='-', connectionstyle=__annot_con_style(angle, to_node_angle),
                            linestyle=':')
        areas_patches.append(a)

    # edges between networks
    edges_patches = []
    for (n1, n2), d in sub_g.edges.items():
        e = FancyArrowPatch(posA=nodes_coords[n1], posB=nodes_coords[n2],arrowstyle='-',
                            connectionstyle=__edge_con_style(nodes_angles[n1], nodes_angles[n2], bending=edges_bending),
                            linewidth=np.abs(d['correlation'])*4, color=cmap(d['correlation']/2+0.5),
                            alpha=np.abs(d['correlation']))
        edges_patches.append(e)

    return networks_markers, areas_patches, edges_patches


def _spatial_plot_background(graph: SpatioTemporalGraph, ax: Axes = None) -> None:
    """Plot the background of a spatial plot.

    Parameters
    ----------
    graph: SpatioTemporalGraph
        The graph to plot.
    ax: matplotlib.axes.Axes, optional
        The axes on which to plot. If not set, the current axes will be used.
    """
    if ax is None:
        ax = plt.gca()
    ax.axis('off')

    rels, angles, x_areas, y_areas = __areas_positions(graph)
    regions = rels['Name_Region'].unique()
    n = len(rels)

    # plot regions in a pie
    regions_cmap = cm.get_cmap('tab20')
    ax.pie([len(rels[rels['Name_Region'] == region]) / n
            for region in regions],
           radius=2.25, startangle=-360 / n / 2,
           labels=regions, labeldistance=1.1, rotatelabels=False,
           colors=[regions_cmap(i) for i in range(len(regions))],
           wedgeprops=dict(width=1, edgecolor='w', alpha=0.3))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    # plot areas' labels
    for i, (x_area, y_area, area) in enumerate(zip(x_areas, y_areas, rels['Name_Area'])):
        angle = np.rad2deg(angles[i])
        ax.text(x=x_area, y=y_area, s=area,
                va='center', fontsize='x-small', rotation_mode='anchor',
                **__readable_angled_annotation(angle))


def spatial_plot(graph: SpatioTemporalGraph, t: float, ax: Axes = None, edges_bending: float = 3) -> None:
    """Draw a spatial plot for spatio-temporal graph.

    Parameters
    ----------
    graph: SpatioTemporalGraph
        The graph to plot.
    t: float
        The instant to plot in the graph.
    ax: matplotlib.axes.Axes, optional
        The axes on which to plot. If not set, the current axes will be used.
    edges_bending: float, optional
        Controls the bending of the edges. Close to 0 means full bending, close to
        infinity means no bending (default is 3).
    """
    _spatial_plot_background(graph, ax)

    networks_markers, areas_arrows, edges_patches = _spatial_plot_artists(graph, t=t, edges_bending=edges_bending)

    for network_marker in networks_markers:
        ax.add_line(network_marker)

    for area_arrow in areas_arrows:
        ax.add_patch(area_arrow)

    for edge_patch in edges_patches:
        ax.add_patch(edge_patch)


class __CoordinatesGenerator:
    """Utility to generate temporal paths from a spatio-temporal graph."""

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
        """Generate the coordinates of the temporal paths.

        Parameters
        ----------
        nodes: list[int]
            The nodes starting the paths.
        base_y: int
            The initial height location of the path.

        Returns
        -------
        dict[int, tuple[int, int]]
            A dictionary mapping a node to its time/height coordinates.
        """
        self.__max_heights = dict()
        self.__coords = {n: (self.g.nodes[n]['t'], base_y + i) for i, n in enumerate(nodes)}

        for i, node in enumerate(nodes):
            trans_list = self.__next_temp_trans(node)
            self.__generate_coords_for_path_rec(trans_list, base_y + i)

        return self.__coords


@cache
def _trans_color(transition: RC5) -> str:
    """Defines the color to use for a given RC5 transition.

    Parameters
    ----------
    transition: RC5
        The transition between two nodes.

    Returns
    -------
    str
        The transition's color name.
    """
    if transition == RC5.PP:
        return 'red'
    elif transition == RC5.PPi:
        return 'blue'
    elif transition == RC5.PO:
        return 'limegreen'
    else:
        return 'black'


class __PathDrawer:
    """Utility to draw temporal paths of a spatio-temporal graph on an axis."""

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
        """Draw the temporal paths.

        Parameters
        ----------
        coords: dict[int, tuple[int, int]]
            The coordinates of the nodes in the temporal paths.
        nodes: list[int]
            The nodes starting the paths.
        base_y: int
            The initial height location of the path.
        """
        self.__done = set()
        self.__lines = list()
        self.__colors = list()

        for i, node in enumerate(nodes):
            trans_list = self.__next_temp_trans(node)
            self.__draw_rec(coords, trans_list, node, self.g.nodes[node]['t'], base_y + i)

        self.axe.add_collection(
            LineCollection(self.__lines, colors=self.__colors,
                           linewidths=1.5, linestyles='-'))


def temporal_plot(graph: SpatioTemporalGraph, ax: Axes = None) -> None:
    """Draw a temporal plot for a spatio-temporal graph.

    Parameters
    ----------
    graph: SpatioTemporalGraph
        The spatio-temporal graph.
    ax: matplotlib.axes.Axes, optional
        The axes on which to plot.  If not set, the current axes will be used.
    """
    if ax is None:
        ax = plt.gca()

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

        colors = [graph.nodes[n]['internal_strength'] for n in coords.keys()]
        ax.scatter(*list(zip(*coords.values())), zorder=2.1,
                s=30, c=colors, cmap=cmap, edgecolors='k', linewidths=1, vmin=-1, vmax=1)

        heights.append(max(coords.values(), key=lambda x: x[1])[1] + 1 - y)
        y += heights[-1] + 1

    # draw limits of regions
    o = 0
    regions_cmap = cm.get_cmap('tab20')
    ticks = []
    for r, _ in enumerate(regions):
        m = heights[r]
        o += m
        ax.fill_between([times.min() - 0.5, times.max() + 0.5], o - m - 1, o,
                        fc=regions_cmap(r), alpha=0.2)
        ticks.append((o - m - 1 + o) / 2)
        o += 1

    # set up the axes
    ax.spines[['left', 'top', 'right']].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.set_xlabel("Time")
    ax.set_xlim(times.min()-1, times.max()+1)

    ax.set_yticks(ticks, regions)
    for tick in ax.get_yaxis().get_major_ticks():
        tick.tick1line.set_visible(False)
    ax.set_ylim(-1, sum(heights) + len(heights) - 1)


def _inch2cm(inch: float) -> float:
    return inch / 2.54


def _calc_limits(t: int, w: int, limits: tuple[int, int]) -> tuple[float, float]:
    half_w = w // 2
    left, right = t - half_w,  t + half_w
    if left <= limits[0]:
        return limits[0] - 0.5, min(limits[0] + w, limits[1]) + 0.5
    elif limits[1] <= right:
        return max(limits[1] - w, limits[0]) - 0.5, limits[1] + 0.5
    else:
        return left - 0.5, right + 0.5


class DynamicTimeCursor(Cursor):
    """A dynamic cursor for time points."""

    def __init__(self, axe: Axes, func: Callable[[int], None], **lineprops):
        super().__init__(ax=axe, horizOn=False, useblit=True, **lineprops)
        self.__callback = func
        self.__last_t = None

    def onmove(self, event):
        if self.ignore(event):
            return
        if not self.canvas.widgetlock.available(self):
            return

        if event.inaxes == self.ax:
            t = int(round(event.xdata))

            if self.__last_t != t:
                self.__callback(t)
                self.__last_t = t
            event.xdata = t

        super().onmove(event)


@dataclass
class DynamicPlot:
    """A dynamic plot that contains both temporal and spatial plot with interactivity."""
    graph: SpatioTemporalGraph
    size: float
    time_window: int = None

    @property
    def __networks_markers(self) -> list[Line2D]:
        return list(self.spl_axe.lines)

    @property
    def __areas_annotations(self) -> list[Annotation]:
        return [t for t in self.spl_axe.texts if isinstance(t, Annotation)]

    @property
    def __edges_patches(self) -> list[FancyArrowPatch]:
        return [p for p in self.spl_axe.patches if isinstance(p, FancyArrowPatch)]

    def __create_figure(self) -> None:
        self.fig = plt.figure(figsize=(_inch2cm(self.size), _inch2cm(self.size / 3)), layout='constrained')
        gs = GridSpec(nrows=1, ncols=2, figure=self.fig, width_ratios=[2, 1])
        self.tpl_axe = self.fig.add_subplot(gs[0])
        self.spl_axe = self.fig.add_subplot(gs[1])
        self.fig.spl_bkd = None

    def __on_window_resized(self) -> None:
        self.fig.spl_bkd = None

    def __initialize_figure(self) -> None:
        # define initial time to show and limits of temporal plot
        init_t = 0
        limits_t = self.graph.graph['min_time'], self.graph.graph['max_time']
        if self.time_window is None:
            self.time_window = limits_t[1]

        # plot both temporal and spatial plots
        temporal_plot(self.graph, ax=self.tpl_axe)
        _spatial_plot_background(self.graph, ax=self.spl_axe)
        self.tpl_axe.set_xlim(*_calc_limits(init_t, self.time_window, limits_t))

        # set the initial spatial plot display
        self.__on_cursor_changed(init_t)

        # capture resize events (update background of spatial plot)
        # triggers the reset of the spatial axe background
        self.fig.canvas.mpl_connect('resize_event', lambda e: self.__on_window_resized())

    def __remove_artists(self, artists: list[Artist]) -> None:
        for a in artists:
            a.remove()

    def __add_artists(self, artists: list[Artist], adding_func: Callable[[Artist], None]) -> None:
        for a in artists:
            adding_func(a)
            self.spl_axe.draw_artist(a)

    def __modify_artists(self, old_artists: list[Artist], new_artists: list[Artist],
                         modify_func: Callable[[Artist, Artist], None]) -> None:
        for ao, an in zip(old_artists, new_artists):
            modify_func(ao, an)
            self.spl_axe.draw_artist(ao)

    @staticmethod
    def __modify_networks_markers(ol: Line2D, nl: Line2D) -> None:
        ol.set(data=nl.get_data(), mfc=nl.get_markerfacecolor(), ms=nl.get_markersize())

    @staticmethod
    def __modify_edges_patches(olp: FancyArrowPatch, nwp: FancyArrowPatch) -> None:
        vx = nwp.get_path().vertices
        olp.set_positions(posA=(float(vx[0][0]), float(vx[0][1])), posB=(float(vx[-1][0]), float(vx[-1][1])))
        olp.set(fc=nwp.get_fc(), ec=nwp.get_ec(), alpha=nwp.get_alpha(), lw=nwp.get_linewidth(),
                connectionstyle=nwp.get_connectionstyle())

    def __update_artists(self, old_artists_lists: list[list[Artist]], new_artists_lists: list[list[Artist]],
                         modifiers: list[Callable[[Artist, Artist], None]], adders: list[Callable[[Artist], None]]):
        # remove excess of old artists (and remember the initial count)
        no = []
        for oal, nal in zip(old_artists_lists, new_artists_lists):
            no.append(len(oal))
            self.__remove_artists(oal[len(nal):])

        # restore the background
        self.fig.canvas.restore_region(self.fig.spl_bkd)

        # modify reusable artists
        for oal, nal, mod in zip(old_artists_lists, new_artists_lists, modifiers):
            self.__modify_artists(oal, nal, mod)

        # add excess of new artists
        for nal, n, add in zip(new_artists_lists, no, adders):
            self.__add_artists(nal[n:], add)

    def __recreate_artists(self, old_artists_lists: list[list[Artist]], new_artists_lists: list[list[Artist]],
                           adders: list[Callable[[Artist], None]]):
        # remove all old artists
        for oal in old_artists_lists:
            self.__remove_artists(oal)

        # save a new background for spatial axe
        self.fig.canvas.draw()
        self.fig.spl_bkd = self.fig.canvas.copy_from_bbox(self.spl_axe.bbox)

        # add all new artists
        for nal, add in zip(new_artists_lists, adders):
            self.__add_artists(nal, add)

    def __on_cursor_changed(self, t: int) -> None:
        # get old/new artists
        old_networks_markers = self.__networks_markers
        old_areas_annotations = self.__areas_annotations
        old_edges_patches = self.__edges_patches

        new_networks_markers, new_areas_patches, new_edges_patches = _spatial_plot_artists(self.graph, t=t)

        # update or recreate artists (if the background has been invalidated)
        old_artists = [old_edges_patches, old_networks_markers]
        new_artists = [new_edges_patches, new_networks_markers]
        adders = [self.spl_axe.add_patch, self.spl_axe.add_line]

        if self.fig.spl_bkd is None:
            self.__recreate_artists(old_artists, new_artists, adders)
        else:
            modifiers = [DynamicPlot.__modify_edges_patches, DynamicPlot.__modify_networks_markers]
            self.__update_artists(old_artists, new_artists, modifiers, adders)

        # blit the spatial axes to draw only changed artists
        self.fig.canvas.blit(self.spl_axe.bbox)

    def __initialize_widgets(self) -> None:
        self.fig.t_cursor = DynamicTimeCursor(self.tpl_axe, self.__on_cursor_changed,
                                              color='k', lw=0.8, ls='--')

    def plot(self):
        self.__create_figure()
        self.__initialize_figure()
        self.__initialize_widgets()
