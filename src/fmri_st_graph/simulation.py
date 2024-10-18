"""Defines the tools to simulate spatio-temporal graphs and functional connectivity data."""
from dataclasses import dataclass
from itertools import combinations

import networkx as nx
import numpy as np

from .graph import SpatioTemporalGraph


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
            combs_size = len(combs)

            for _ in range(min(n, combs_size)):
                elem = combs.pop(self.rng.integers(combs_size))
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
    def __init__(self, graph_struct: SpatioTemporalGraph, threshold: float = 0.4) -> None:
        self.graph_struct = graph_struct
        self.threshold = threshold

        self.__rng = np.random.default_rng()
        self.__network_edges_filler = _CorrelationMatrixNetworksEdgesFiller(self.threshold, self.__rng)
        self.__inter_region_edges_filler = _CorrelationMatrixInterRegionEdgesFiller(self.threshold, self.__rng)

        self.__init_validation__()

    def __init_validation__(self):
        if self.threshold < 0 or self.threshold > 1:
            raise ValueError("The threshold must be within range [0, 1]!")

    def __simulate_corr_matrix(self, spatial_graph: nx.DiGraph) -> np.array:
        matrix = np.eye(len(self.graph_struct.areas))

        self.__network_edges_filler.fill(spatial_graph, matrix)
        self.__inter_region_edges_filler.fill(spatial_graph, matrix)

        null_elements = matrix == 0
        bound = self.threshold * 0.99
        matrix[null_elements] = self.__rng.uniform(low=-bound, high=bound, size=null_elements.sum())

    def simulate(self) -> np.array:
        return np.array([self.__simulate_corr_matrix(self.graph_struct.subgraph(t=t))
                         for t in self.graph_struct.time_range])
