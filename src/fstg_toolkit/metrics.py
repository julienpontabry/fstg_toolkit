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

import multiprocessing
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Callable, Sequence

import networkx as nx
import numpy as np
import pandas as pd

from .app.core.io import GraphsDataset
from .graph import SpatioTemporalGraph, RC5

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


metrics_registry: dict[str, MetricsRegistry] = {}


def get_metrics_registry(name: str) -> MetricsRegistry:
    global metrics_registry

    if name not in metrics_registry:
        metrics_registry[name] = MetricsRegistry()

    return metrics_registry[name]


def metrics_index_columns(index_columns: Optional[list[str]]):
    def decorator(func):
        func.index_columns = index_columns
        return func
    return decorator


@metrics_index_columns(['Time'])
def calculate_spatial_metrics(graph: SpatioTemporalGraph) -> list[MetricRecord]:
    registry = get_metrics_registry('local')
    records = []

    for t in graph.time_range:
        g = SpatioTemporalGraph(nx.Graph(graph.sub(t=t)), graph.areas)
        record: dict[str, MetricType] = {'Time': t}

        for name, func in registry:
            record[name] = func(g)

        records.append(record)

    return records


def calculate_temporal_metrics(graph: SpatioTemporalGraph) -> list[MetricRecord]:
    registry = get_metrics_registry('global')
    record: MetricRecord = {}
    g = graph.sub_temporal()

    for name, func in registry:
        record[name] = func(g)

    return [record]


def _process_parallel(subject, dataset, calculator):
    return subject, calculator(dataset.get_graph(subject))


def gather_metrics(dataset: GraphsDataset, selection: Sequence[tuple[str, ...]],
                   calculator: MetricsCalculator, callback: Optional[Callable[[tuple[str, ...]], None]] = lambda s: None,
                   max_cpus: int = multiprocessing.cpu_count() - 1) -> pd.DataFrame:
    n_factors = len(dataset.factors)
    all_records = []
    all_idx = []

    futures = []
    num_workers = min(len(selection), max_cpus)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for subject in selection:
            future = executor.submit(_process_parallel, subject, dataset, calculator)
            futures.append(future)

        for future in as_completed(futures):
            subject, records = future.result()
            all_records += records

            if not isinstance(subject, tuple):
                subject = (subject,)

            all_idx += [subject] * len(records)
            callback(subject)

    df = pd.json_normalize(all_records)
    idx = pd.MultiIndex.from_tuples(all_idx, names=[f'Factor{i + 1}' for i in range(n_factors)] + ['id'])
    df.set_index(idx, inplace=True)

    if hasattr(calculator, 'index_columns'):
        df.set_index(calculator.index_columns, append=True, inplace=True)

    multi_cols = [tuple(c.split('.')) if '.' in c else (c,) for c in df.columns]
    if max([len(e) for e in multi_cols]) > 1:
        df.columns = pd.MultiIndex.from_tuples(multi_cols)
        df.rename(columns={np.nan: ''}, inplace=True)

    return df


def metric(registry_name: str, name: str):
    def decorator(func):
        registry = get_metrics_registry(registry_name)
        registry.add(name, func)
        return func
    return decorator


@metric('local', "Average degree")
def average_degree(graph: SpatioTemporalGraph) -> float:
    return np.mean(nx.degree_histogram(graph))


@metric('local', "Assortativity")
def assortativity(graph: SpatioTemporalGraph) -> float:
    return nx.degree_assortativity_coefficient(graph)


@metric('local', "Clustering coefficient")
def clustering(graph: SpatioTemporalGraph) -> float:
    return nx.average_clustering(graph)


@metric('local', "Global efficiency")
def global_efficiency(graph: SpatioTemporalGraph) -> float:
    return nx.global_efficiency(nx.Graph(graph))


@metric('local', "Density")
def density(graph: SpatioTemporalGraph) -> float:
    return nx.density(graph)


@metric('local', "Modularity")
def modularity(graph: SpatioTemporalGraph) -> float:
    communities = nx.community.greedy_modularity_communities(graph)
    try:
        return nx.community.modularity(graph, communities)
    except ZeroDivisionError:
        return 0.0


# @metric('local', "Mean number of areas")
def mean_areas(graph: SpatioTemporalGraph) -> dict[str, float]:
    return {region: float(np.mean([len(d['areas']) for _, d in graph.sub(region=region).nodes(data=True)]))
            for region in np.unique(graph.areas['Name_Region'])}


@metric('global', "Transitions distribution")
def transitions_distribution(graph: SpatioTemporalGraph) -> dict[str, int]:
    # TODO add distribution per region
    return {trans: len([_ for _, _, d in graph.edges(data=True) if d['transition'] == trans])
            for trans in RC5 if trans != RC5.DC}


@metric('global', "Reorganisation rate")
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


@metric('global', "Burstiness")
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


@metric('global', "Memory")
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
