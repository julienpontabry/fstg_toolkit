from dataclasses import dataclass, field
from typing import Optional, Callable, Iterable

import networkx as nx
import numpy as np
import pandas as pd

from .graph import SpatioTemporalGraph, RC5
from .app.core.io import GraphsDataset


type MetricType = float | list[float] | dict[str, int]
type MetricFunction = Callable[[SpatioTemporalGraph], MetricType]
type MetricRecord = dict[str, MetricType]
type MetricsCalculator = Callable[[SpatioTemporalGraph], list[MetricRecord]]


@dataclass(frozen=True)
class MetricsRegistry:
    __registry: dict[str, MetricFunction] = field(default_factory=lambda: {})

    def add(self, name: str, func: MetricFunction) -> None:
        self.__registry[name] = func

    def remove(self, name: str) -> None:
        del self.__registry[name]

    def __iter__(self):
        return iter(self.__registry.items())


# TODO use a single registry and multiple metrics categories
spatial_metrics_registry: Optional[MetricsRegistry] = None
temporal_metrics_registry: Optional[MetricsRegistry] = None


def get_spatial_metrics_registry() -> MetricsRegistry:
    global spatial_metrics_registry

    if spatial_metrics_registry is None:
        spatial_metrics_registry = MetricsRegistry()

    return spatial_metrics_registry


def get_temporal_metrics_registry() -> MetricsRegistry:
    global temporal_metrics_registry

    if temporal_metrics_registry is None:
        temporal_metrics_registry = MetricsRegistry()

    return temporal_metrics_registry


def metrics_index_columns(index_columns: Optional[list[str]]):
    def decorator(func):
        func.index_columns = index_columns
        return func
    return decorator


@metrics_index_columns(['Time'])
def calculate_spatial_metrics(graph: SpatioTemporalGraph) -> list[MetricRecord]:
    registry = get_spatial_metrics_registry()
    records = []

    for t in graph.time_range:
        g = SpatioTemporalGraph(nx.Graph(graph.sub(t=t)), graph.areas)
        record: dict[str, MetricType] = {'Time': t}

        for name, func in registry:
            record[name] = func(g)

        records.append(record)

    return records


def calculate_temporal_metrics(graph: SpatioTemporalGraph) -> list[MetricRecord]:
    registry = get_temporal_metrics_registry()
    record: MetricRecord = {}
    g = graph.sub_temporal()

    for name, func in registry:
        record[name] = func(g)

    return [record]


def gather_metrics(dataset: GraphsDataset, selection: Iterable[tuple[str, ...]],
                   calculator: MetricsCalculator) -> pd.DataFrame:
    n_factors = len(dataset.factors)
    all_records = []
    all_idx = []

    for subject in selection:
        records = calculator(dataset.get_graph(subject))
        all_records += records
        all_idx += [subject] * len(records)

    df = pd.json_normalize(all_records)
    idx = pd.MultiIndex.from_tuples(all_idx, names=[f'factor{i + 1}' for i in range(n_factors)] + ['id'])
    df.set_index(idx, inplace=True)

    if hasattr(calculator, 'index_columns'):
        df.set_index(calculator.index_columns, append=True, inplace=True)

    multi_cols = [tuple(c.split('.')) if '.' in c else (c,) for c in df.columns]
    if max([len(e) for e in multi_cols]) > 1:
        df.columns = pd.MultiIndex.from_tuples(multi_cols)
        df.rename(columns={np.nan: ''}, inplace=True)

    return df


def spatial_metric(name):
    def decorator(func):
        calculator = get_spatial_metrics_registry()
        calculator.add(name, func)
        return func
    return decorator


def temporal_metric(name):
    def decorator(func):
        calculator = get_temporal_metrics_registry()
        calculator.add(name, func)
        return func
    return decorator


@spatial_metric("Average degree")
def average_degree(graph: SpatioTemporalGraph) -> float:
    return np.mean(nx.degree_histogram(graph))


@spatial_metric("Assortativity")
def assortativity(graph: SpatioTemporalGraph) -> float:
    return nx.degree_assortativity_coefficient(graph)


@spatial_metric("Clustering coefficient")
def clustering(graph: SpatioTemporalGraph) -> float:
    return nx.average_clustering(graph)


@spatial_metric("Global efficiency")
def global_efficiency(graph: SpatioTemporalGraph) -> float:
    return nx.global_efficiency(nx.Graph(graph))


@spatial_metric("Density")
def density(graph: SpatioTemporalGraph) -> float:
    return nx.density(graph)


@spatial_metric("Modularity")
def modularity(graph: SpatioTemporalGraph) -> float:
    communities = nx.community.greedy_modularity_communities(graph)
    return nx.community.modularity(graph, communities)


# @spatial_metric("Mean number of areas")
def mean_areas(graph: SpatioTemporalGraph) -> dict[str, float]:
    return {region: float(np.mean([len(d['areas']) for _, d in graph.sub(region=region).nodes(data=True)]))
            for region in np.unique(graph.areas['Name_Region'])}


@temporal_metric("Transitions distribution")
def transitions_distribution(graph: SpatioTemporalGraph) -> dict[str, int]:
    # TODO add distribution per region
    return {trans: len([_ for _, _, d in graph.edges(data=True) if d['transition'] == trans])
            for trans in list(RC5) if trans != RC5.DC}


@temporal_metric("Reorganisation rate")
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


@temporal_metric("Burstiness")
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


@temporal_metric("Memory")
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
