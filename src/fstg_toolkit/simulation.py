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

"""Defines the tools to simulate spatio-temporal graphs and functional connectivity data."""
from dataclasses import dataclass
from functools import reduce
from itertools import combinations
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
from networkx.classes.reportviews import NodeView

from .graph import RC5, SpatioTemporalGraph, subgraph_nodes


def _fill_matrix(connections, correlations, matrix):
    for (e1, e2), corr in zip(connections, correlations):
        i = e1 - 1
        j = e2 - 1
        matrix[i, j] = matrix[j, i] = corr


@dataclass(frozen=True)
class _CorrelationMatrixNetworksEdgesFiller:
    threshold: float
    rng: np.random.Generator

    def __choose_connections(self, network: list[int]) -> list[tuple[int, int]]:
        def __new_graph(trial_connections: list[tuple[int, int]]) -> nx.Graph:
            g = nx.Graph()
            g.add_nodes_from(network)
            g.add_edges_from(trial_connections)
            return g

        def __try_random_connections(net: list[int]) -> list[tuple[int, int]]:
            combs = list(combinations(net, 2))

            if len(combs) == 1:
                return combs
            else:
                selected = []

                for _ in range(self.rng.integers(low=len(network) - 1, high=len(combs))):
                    elem = combs.pop(self.rng.integers(len(combs)))
                    selected.append(elem)

                return selected

        trial_graph = __new_graph(__try_random_connections(network))

        while not nx.is_connected(trial_graph):
            trial_graph = __new_graph(__try_random_connections(network))

        return list(trial_graph.edges)

    def __mean_corr_sampler(self, size: int, mean: float) -> list[float]:
        def __sample(low: float, high: float) -> np.array:
            values = self.rng.uniform(low=low, high=high, size=size)
            return values + mean - values.mean()

        rad = abs(mean) - self.threshold
        a, b = mean - rad, mean + rad

        samples = __sample(a, b)
        while any(samples < a) or any(samples > b):
            samples = __sample(a, b)

        return samples.tolist()

    def fill(self, spatial_graph: nx.DiGraph, matrix: np.array) -> None:
        networks = [(data['areas'], data['internal_strength'])
                    for _, data in spatial_graph.nodes.items()]

        for network, mean_corr in networks:
            connections = self.__choose_connections(network)
            correlations = self.__mean_corr_sampler(len(connections), mean_corr)
            _fill_matrix(connections, correlations, matrix)


@dataclass(frozen=True)
class _CorrelationMatrixInterRegionEdgesFiller:
    threshold: float
    rng: np.random.Generator

    def __max_correlation_sampler(self, size: int, target: float) -> list[float]:
        sample_fun = np.max if target >= 0 else np.min

        def __sample(low: float, high: float) -> np.array:
            values = self.rng.uniform(low=low, high=high, size=size)
            return values + target - sample_fun(values)

        thr = np.sign(target) * self.threshold
        mean = (thr + target) / 2
        rad = abs(target - mean)
        a, b = mean - rad, mean + rad

        samples = __sample(a, b)
        while any(samples < a) or any(samples > b):
            samples = __sample(a, b)

        return samples

    def __choose_inter_region_connections(self, network1: set, network2: set) -> list[tuple[int, int]]:
        def __choose_inter_region_connection(k: int, network: set, n: int) -> list[tuple[int, int]]:
            combs = list(zip([k]*len(network), network))
            selected = []

            for _ in range(min(n, len(combs))):
                elem = combs.pop(self.rng.integers(len(combs)))
                selected.append(elem)

            return selected

        connections_sizes = self.rng.poisson(lam=1, size=len(network1))
        connections = []

        for node, nb_connections in zip(network1, connections_sizes):
            connections += __choose_inter_region_connection(node, network2, max(1, nb_connections))

        return connections

    def fill(self, spatial_graph: nx.DiGraph, matrix: np.array) -> None:
        for (node1, node2), data in spatial_graph.edges.items():
            connections = self.__choose_inter_region_connections(
                spatial_graph.nodes[node1]['areas'], spatial_graph.nodes[node2]['areas'])
            correlations = self.__max_correlation_sampler(len(connections), data['correlation'])
            _fill_matrix(connections, correlations, matrix)


class CorrelationMatrixSequenceSimulator:
    """Simulate a sequence of correlation matrices from a spatio-temporal graph.

    Examples
    --------
    >>> graph = nx.DiGraph()
    >>> graph.add_node(1, t=0, areas={1, 2}, region='Region 1', internal_strength=0.98)
    >>> graph.add_node(2, t=0, areas={3, 4}, region='Region 2', internal_strength=-0.98)
    >>> graph.add_edge(1, 2, correlation=0.94, t=0, type='spatial')
    >>> graph.add_edge(2, 1, correlation=0.94, t=0, type='spatial')
    >>> graph.graph['min_time'] = 0
    >>> graph.graph['max_time'] = 0
    >>> areas = pd.DataFrame({'Id_Area': [1, 2, 3, 4],
    ...                       'Name_Area': ['A1', 'A2', 'A3', 'A4'],
    ...                       'Name_Region': ['R1', 'R1', 'R2', 'R2']})
    >>> areas.set_index('Id_Area', inplace=True)
    >>> simulator = CorrelationMatrixSequenceSimulator(SpatioTemporalGraph(graph, areas), threshold=0.4,
    ...                                                rng=np.random.default_rng(40))
    >>> matrix = simulator.simulate()
    >>> matrix.shape
    (1, 4, 4)
    >>> matrix
    array([[[ 1.        ,  0.98      ,  0.65453818,  0.94      ],
            [ 0.98      ,  1.        ,  0.85381682,  0.61873641],
            [ 0.65453818,  0.85381682,  1.        , -0.98      ],
            [ 0.94      ,  0.61873641, -0.98      ,  1.        ]]])
    """

    def __init__(self, graph: SpatioTemporalGraph, threshold: float = 0.4,
                 rng: np.random.Generator = np.random.default_rng()) -> None:
        self.graph = graph
        self.threshold = threshold

        self.__rng = rng
        self.__network_edges_filler = _CorrelationMatrixNetworksEdgesFiller(self.threshold, self.__rng)
        self.__inter_region_edges_filler = _CorrelationMatrixInterRegionEdgesFiller(self.threshold, self.__rng)

        self.__init_validation__()

    def __init_validation__(self):
        if self.threshold < 0 or self.threshold > 1:
            raise ValueError("The threshold must be within range [0, 1]!")

    def __simulate_corr_matrix(self, spatial_graph: nx.DiGraph) -> np.array:
        matrix = np.eye(len(self.graph.areas))

        self.__network_edges_filler.fill(spatial_graph, matrix)
        self.__inter_region_edges_filler.fill(spatial_graph, matrix)

        null_elements = matrix == 0
        bound = self.threshold * 0.99
        matrix[null_elements] = self.__rng.uniform(low=-bound, high=bound, size=null_elements.sum())

        return matrix

    def simulate(self) -> np.array:
        """Simulate the sequence of correlation matrices.

        Returns
        -------
        numpy.array
            A 3D-shaped array that contains the correlations matrices for each time.
        """
        return np.array([self.__simulate_corr_matrix(self.graph.sub(t=t))
                         for t in self.graph.time_range])


def __trans(sources: Iterable[int] | int, targets: Iterable[int] | int, kind: str) -> list[tuple[int, int, RC5]]:
    if kind.lower() == 'split':
        return [(sources, target, RC5.PPi) for target in targets]
    elif kind.lower() == 'merge':
        return [(source, targets, RC5.PP) for source in sources]
    else:
        trans = RC5.EQ if kind.lower() == 'eq' else RC5.PO
        return [(sources, targets, trans)]


def __def2areas(areas_def: tuple[int, int] | Iterable[int] | int) -> set[int]:
    if isinstance(areas_def, tuple) and len(areas_def) == 2:
        start, end = areas_def
        return set(range(start, end + 1))
    elif isinstance(areas_def, Iterable):
        return set(areas_def)
    else:
        return {areas_def}


def generate_pattern(networks_list: list[list[tuple[tuple[int, int], int, float]]],
                     spatial_edges: list[tuple[int, int, float]],
                     temporal_edges: list[tuple[Iterable[int] | int, Iterable[int] | int, str]]) -> SpatioTemporalGraph:
    """Generate a pattern with the specified properties.

    Parameters
    ----------
    networks_list: list[list[tuple[tuple[int, int], int, float]]]
        A list of nodes per time instant, defined themselves by a tuple of area range, region id and
        internal strength.
    spatial_edges: list[tuple[int, int, float]]
        A list of spatial edges defined by a tuple of source/target nodes and a correlation.
    temporal_edges: list[tuple[Iterable[int] | int, Iterable[int] | int, str]]
        A list of temporal edges, defined by a tuple of source(s)/target(s) nodes and a transition.

    Returns
    -------
    SpatioTemporalGraph
        A spatio-temporal graph that can be used as a pattern.

    Example
    -------
    >>> pattern = generate_pattern(
    ...     networks_list=[[((1, 5), 1, -0.2), ((6, 7), 2, 0.3), ((8, 10), 2, 0.6)],
    ...                    [((1, 5), 1, 0.6), ((6, 10), 2, -0.5)]],
    ...     spatial_edges=[(1, 2, 0.45), (4, 5, 0.8)],
    ...     temporal_edges=[(1, 4, 'eq'), ((2, 3), 5, 'merge')])
    >>> pattern.nodes
    NodeView((1, 2, 3, 4, 5))
    >>> pattern.edges
    OutEdgeView([(1, 2), (1, 4), (2, 1), (2, 5), (3, 5), (4, 5), (5, 4)])
    >>> pattern.areas
            Name_Area Name_Region
    Id_Area
    1          Area 1    Region 1
    2          Area 2    Region 1
    3          Area 3    Region 1
    4          Area 4    Region 1
    5          Area 5    Region 1
    6          Area 6    Region 2
    7          Area 7    Region 2
    8          Area 8    Region 2
    9          Area 9    Region 2
    10        Area 10    Region 2
    """
    g = nx.DiGraph()
    g.graph['min_time'] = 0
    g.graph['max_time'] = len(networks_list) - 1

    k = 1
    all_areas = set()
    areas_regions = {}
    for t, networks in enumerate(networks_list):
        for areas_def, region_id, strength in networks:
            areas = __def2areas(areas_def)
            all_areas |= areas

            region = f"Region {region_id}"
            for area in areas:
                areas_regions[area] = region

            g.add_node(k, t=t, areas=areas, region=region, internal_strength=strength)
            k += 1

    for source, target, corr in spatial_edges:
        g.add_edge(source, target, correlation=corr, type='spatial')
        g.add_edge(target, source, correlation=corr, type='spatial')

    for temporal_link in temporal_edges:
        for source, target, rc5 in __trans(*temporal_link):
            g.add_edge(source, target, transition=rc5, type='temporal')

    all_areas = sorted(all_areas)
    areas = pd.DataFrame({'Id_Area': all_areas,
                          'Name_Area': [f"Area {a}" for a in all_areas],
                          'Name_Region': [areas_regions[a] for a in all_areas]})
    areas.set_index('Id_Area', inplace=True)

    return SpatioTemporalGraph(g, areas)


class SpatioTemporalGraphSimulator:
    """Simulator for spatio-temporal graphs.

    The simulator needs predefined patterns (created either manually or
    automatically) to generate a full spatio-temporal graph with those patterns
    included as instructed, eventually with in-between repeats.

    Examples
    --------
    >>> pattern1 = generate_pattern(
    ...     networks_list=[
    ...         [((1, 2), 1, 0.7), (3, 1, 1), ((4, 5), 2, -0.8)],
    ...         [((1, 3), 1, 0.8), ((4, 5), 2, -0.8)]],
    ...     spatial_edges=[(1, 3, 0.5), (4, 5, 0.6)],
    ...     temporal_edges=[((1, 2), 4, 'merge'), (3, 5, 'eq')])
    >>> pattern2 = generate_pattern(
    ...     networks_list=[
    ...         [((1, 3), 1, 0.8), ((4, 5), 2, -0.8)],
    ...         [((1, 2), 1, 0.7), (3, 1, 1), ((4, 5), 2, -0.8)]],
    ...     spatial_edges=[(1, 2, 0.6), (3, 5, 0.5)],
    ...     temporal_edges=[(1, (3, 4), 'split'), (2, 5, 'eq')])
    >>> simulator = SpatioTemporalGraphSimulator(p1=pattern1, p2=pattern2)
    >>> graph = simulator.simulate('p2', 3, 'p1')
    >>> graph.nodes
    NodeView((1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19))
    >>> graph.edges
    OutEdgeView([(1, 2), (1, 3), (1, 4), (2, 1), (2, 5), (3, 5), (3, 6), (4, 7), (5, 3), (5, 8),
    (6, 8), (6, 9), (7, 10), (8, 6), (8, 11), (9, 11), (9, 12), (10, 13), (11, 9), (11, 14), (12, 14),
    (12, 15), (13, 16), (14, 12), (14, 17), (15, 17), (15, 18), (16, 18), (17, 15), (17, 19), (18, 19), (19, 18)])
    """
    def __init__(self, **patterns: SpatioTemporalGraph) -> None:
        self.__patterns = patterns

    def _simulate_areas_descriptions(self, patterns: list[str | int]) -> pd.DataFrame:
        areas_descriptions = [self.__patterns[pattern].areas
                              for pattern in patterns
                              if isinstance(pattern, str)]
        return pd.concat(areas_descriptions).drop_duplicates()

    @staticmethod
    def __shift_node_data(data: dict[str, any], dt: int) -> dict[str, any]:
        tmp = dict(data)
        tmp['t'] += dt
        return tmp

    @staticmethod
    def __shift_nodes(nodes: NodeView, dt: int, k: int) -> list[tuple[int, dict[str, any]]]:
        return [(n + k, SpatioTemporalGraphSimulator.__shift_node_data(d, dt))
                for n, d in sorted(nodes.items(), key=lambda x: x[0])]

    def _simulate_graph_from_patterns(self, patterns: list[str | int]) -> nx.DiGraph:
        g = nx.DiGraph(self.__patterns[patterns[0]])

        for next_pattern in patterns[1:]:
            last_t = g.graph['max_time']
            last_out = subgraph_nodes(g, t=last_t)

            if isinstance(next_pattern, int):
                for i in range(next_pattern):
                    k = len(last_out.nodes)
                    m = (i + 1) * k
                    g.add_nodes_from(SpatioTemporalGraphSimulator.__shift_nodes(last_out.nodes, i + 1, m))
                    g.add_edges_from(reduce(list.__add__, [
                        [(n1 + m, n2 + m, d),
                         (n2 + m, n1 + m, d)]
                         for (n1, n2), d in last_out.edges.items()
                         if d['type'] == 'spatial'], []))
                    g.add_edges_from([(n + i * k, n + m, {'transition': RC5.EQ, 'type': 'temporal'})
                                      for n in sorted(last_out.nodes)])

                g.graph['max_time'] += next_pattern
            elif isinstance(next_pattern, str):
                next_pattern = self.__patterns[next_pattern]
                k = max(g.nodes)
                dt = g.graph['max_time'] - g.graph['min_time'] + 1

                # add pattern (with time and nodes shifted appropriately)
                g.add_nodes_from(SpatioTemporalGraphSimulator.__shift_nodes(next_pattern.nodes, dt, k))
                g.add_edges_from([(e1 + k, e2 + k, d)
                                  for (e1, e2), d in next_pattern.edges.items()])
                g.graph['max_time'] += next_pattern.graph['max_time'] + 1

                # make the connection between last pattern and next one
                next_in = subgraph_nodes(next_pattern, t=next_pattern.graph['min_time'])

                for nout, nin in zip(sorted(last_out.nodes), sorted(next_in.nodes)):
                    g.add_edge(nout, nin + k, transition=RC5.EQ, type='temporal')
            else:
                raise ValueError(f"pattern type {type(next_pattern)} "
                                 "is not recognized! It must be either "
                                 "int or nx.DiGraph.")

        return g

    def simulate(self, *patterns: str | int) -> SpatioTemporalGraph:
        """Simulate the given sequence of pattern.

        Parameters
        ----------
        patterns: tuple[str | int]
            The sequence of patterns. A string references a pattern registered at the
            creation of the simulator and an integer reference a number of times repeats
            in-between patterns. A repeat is a subgraph of the last time of the last pattern.

        Returns
        -------
        SpatioTemporalGraph
            The built spatio-temporal graph.
        """
        return SpatioTemporalGraph(self._simulate_graph_from_patterns(list(patterns)),
                                   self._simulate_areas_descriptions(list(patterns)))