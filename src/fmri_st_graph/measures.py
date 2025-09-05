from dataclasses import dataclass, field
from typing import Optional, Callable, Any

import networkx as nx
import numpy as np
import pandas as pd

from .graph import SpatioTemporalGraph, RC5

type MeasureFunction = Callable[[SpatioTemporalGraph], float|list[float]]


@dataclass(frozen=True)
class MeasuresCalculator:
    __spatial_measures_registry: dict[str, MeasureFunction] = field(default_factory=lambda: {})
    __temporal_measures_registry: dict[str, MeasureFunction] = field(default_factory=lambda: {})

    def add_spatial(self, name: str, func: MeasureFunction) -> None:
        self.__spatial_measures_registry[name] = func

    def remove_spatial(self, name: str) -> None:
        del self.__spatial_measures_registry[name]

    def list_spatial(self) -> list[str]:
        return list(self.__spatial_measures_registry.keys())

    def calculate_spatial_measures(self, graph: SpatioTemporalGraph) -> pd.DataFrame:
        records = []
        # FIXME also include idx in the records

        for t in graph.time_range:
            g = SpatioTemporalGraph(nx.Graph(graph.sub(t=t)), graph.areas)
            record: dict[str, Any] = {'t': t}
            for name, func in self.__spatial_measures_registry.items():
                record[name] = func(g)
            records.append(record)

        return pd.DataFrame.from_records(records)

    def add_temporal(self, name: str, func: MeasureFunction) -> None:
        self.__temporal_measures_registry[name] = func

    def remove_temporal(self, name: str) -> None:
        del self.__temporal_measures_registry[name]

    def list_temporal(self) -> list[str]:
        return list(self.__temporal_measures_registry.keys())

    def calculate_temporal_measures(self, graph: SpatioTemporalGraph) -> pd.DataFrame:
        g = graph.sub_temporal()
        pass  # TODO implement


singleton_measure_calculator: Optional[MeasuresCalculator] = None


def get_measures_calculator() -> MeasuresCalculator:
    global singleton_measure_calculator

    if singleton_measure_calculator is None:
        singleton_measure_calculator = MeasuresCalculator()

    return singleton_measure_calculator


def spatial_measure(name):
    def decorator(func):
        calculator = get_measures_calculator()
        calculator.add_spatial(name, func)
        return func
    return decorator


def temporal_measure(name):
    def decorator(func):
        calculator = get_measures_calculator()
        calculator.add_temporal(name, func)
        return func
    return decorator


@spatial_measure("Average degree")
def average_degree(graph: SpatioTemporalGraph) -> float:
    return np.mean(nx.degree_histogram(graph))


@spatial_measure("Assortativity")
def assortativity(graph: SpatioTemporalGraph) -> float:
    return nx.degree_assortativity_coefficient(graph)


@spatial_measure("Clustering coefficient")
def clustering(graph: SpatioTemporalGraph) -> float:
    return nx.average_clustering(graph)


@spatial_measure("Global efficiency")
def global_efficiency(graph: SpatioTemporalGraph) -> float:
    return nx.global_efficiency(graph)


@spatial_measure("Density")
def density(graph: SpatioTemporalGraph) -> float:
    return nx.density(graph)


@spatial_measure("Modularity")
def modularity(graph: SpatioTemporalGraph) -> float:
    communities = nx.community.greedy_modularity_communities(graph)
    return nx.community.modularity(graph, communities)


@spatial_measure("Mean number of areas")
def mean_areas(graph: SpatioTemporalGraph) -> list[float]:
    return [float(np.mean([len(d['areas']) for _, d in graph.sub(region=region)]))
            for region in np.unique(graph.areas['Name_Region'])]


@temporal_measure("Transitions distribution")
def transitions_distribution(graph: SpatioTemporalGraph) -> list[float]:
    pass  # TODO implement

@temporal_measure("Reorganisation rate")
def reorg_rate(graph: SpatioTemporalGraph) -> float:
    pass  # TODO implement

@temporal_measure("Burstiness and memory")
def burstiness_memory(graph: SpatioTemporalGraph) -> list[float]:
    non_eq_edges = [(n1, n2) for n1, n2, d in graph.edges(data=True)
                    if d['transition'] != RC5.EQ]
    event_times = [graph.nodes[n]['t'] for n, _ in non_eq_edges]
    return []  # TODO implement
