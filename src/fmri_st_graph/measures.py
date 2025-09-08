from dataclasses import dataclass, field
from typing import Optional, Callable

import networkx as nx
import numpy as np
import pandas as pd

from .graph import SpatioTemporalGraph, RC5


type MeasureOutput = float | list[float]
type MeasureFunction = Callable[[SpatioTemporalGraph], MeasureOutput]


@dataclass(frozen=True)
class MeasuresRegistry:
    __registry: dict[str, MeasureFunction] = field(default_factory=lambda: {})

    def add(self, name: str, func: MeasureFunction) -> None:
        self.__registry[name] = func

    def remove(self, name: str) -> None:
        del self.__registry[name]

    def __iter__(self):
        return iter(self.__registry.items())


spatial_measures_registry: Optional[MeasuresRegistry] = None
temporal_measures_registry: Optional[MeasuresRegistry] = None


def get_spatial_measures_registry() -> MeasuresRegistry:
    global spatial_measures_registry

    if spatial_measures_registry is None:
        spatial_measures_registry = MeasuresRegistry()

    return spatial_measures_registry


def get_temporal_measures_registry() -> MeasuresRegistry:
    global temporal_measures_registry

    if temporal_measures_registry is None:
        temporal_measures_registry = MeasuresRegistry()

    return temporal_measures_registry



def calculate_spatial_measures(graph: SpatioTemporalGraph) -> pd.DataFrame:
    registry = get_spatial_measures_registry()
    records = []

    for t in graph.time_range:
        g = SpatioTemporalGraph(nx.Graph(graph.sub(t=t)), graph.areas)
        record: dict[str, MeasureOutput] = {'t': t}

        for name, func in registry:
            record[name] = func(g)

        records.append(record)

    return pd.DataFrame.from_records(records)


def calculate_temporal_measures(graph: SpatioTemporalGraph) -> pd.DataFrame:
    registry = get_temporal_measures_registry()
    record: dict[str, MeasureOutput] = {}
    g = graph.sub_temporal()

    for name, func in registry:
        record[name] = func(g)

    return pd.DataFrame.from_records([record])


def spatial_measure(name):
    def decorator(func):
        calculator = get_spatial_measures_registry()
        calculator.add(name, func)
        return func
    return decorator


def temporal_measure(name):
    def decorator(func):
        calculator = get_temporal_measures_registry()
        calculator.add(name, func)
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
    return nx.global_efficiency(nx.Graph(graph))


@spatial_measure("Density")
def density(graph: SpatioTemporalGraph) -> float:
    return nx.density(graph)


@spatial_measure("Modularity")
def modularity(graph: SpatioTemporalGraph) -> float:
    communities = nx.community.greedy_modularity_communities(graph)
    return nx.community.modularity(graph, communities)


@spatial_measure("Mean number of areas")
def mean_areas(graph: SpatioTemporalGraph) -> list[float]:
    return [float(np.mean([len(d['areas']) for _, d in graph.sub(region=region).nodes(data=True)]))
            for region in np.unique(graph.areas['Name_Region'])]


@temporal_measure("Transitions distribution")
def transitions_distribution(graph: SpatioTemporalGraph) -> list[float]:
    # TODO add distribution per region
    return [len([_ for _, _, d in graph.edges(data=True) if d['transition'] == trans])
            for trans in list(RC5)]


@temporal_measure("Reorganisation rate")
def reorg_rate(graph: SpatioTemporalGraph) -> float:
    # TODO add reorganisation rate per region
    nb_temp_edges = len([_ for _, _, d in graph.edges(data=True)])
    nb_temp_noeq_edges = len([_ for _, _, d in graph.edges(data=True) if d['transition'] != RC5.EQ])
    return nb_temp_noeq_edges / nb_temp_edges


class __EventException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class __NoEventException(__EventException):
    def __init__(self):
        super().__init__("No event detected!")


class __NotEnoughEventException(__EventException):
    def __init__(self):
        super().__init__("Not enough event!")


def __interevent(graph: SpatioTemporalGraph) -> tuple[np.ndarray, np.ndarray]:
    non_eq_edges = [(n1, n2) for n1, n2, d in graph.edges(data=True)
                    if d['transition'] != RC5.EQ]
    event_times = [graph.nodes[n]['t'] for n, _ in non_eq_edges]

    if len(event_times) == 0:
        raise __NoEventException()

    t, counts = np.unique(event_times, return_counts=True)

    if len(t) <= 2:
        raise __NotEnoughEventException()

    intervals = np.diff(t)
    weights = counts[1:]

    return intervals, weights


@temporal_measure("Burstiness")
def burstiness(graph: SpatioTemporalGraph) -> float:
    try:
        intervals, weights = __interevent(graph)
        mean = float(np.average(intervals, weights=weights))
        std = float(np.sqrt(np.average((intervals-mean)**2, weights=weights)))
        return (std - mean) / (std + mean)
    except __NoEventException:
        return -1
    except __NotEnoughEventException:
        return 0


@temporal_measure("Memory")
def memory(graph: SpatioTemporalGraph) -> float:
    try:
        intervals, weights = __interevent(graph)

        ti = intervals[:-1]
        mean1 = np.average(ti, weights=weights[:-1])
        std1 = np.sqrt(np.average((ti-mean1)**2, weights=weights[:-1]))

        tip1 = intervals[1:]
        mean2 = np.average(tip1, weights=weights[1:])
        std2 = np.sqrt(np.average((tip1-mean2)**2, weights=weights[1:]))

        return float(np.mean((ti-mean1) * (tip1-mean2)) / (std1 * std2))
    except __EventException:
        return 0
