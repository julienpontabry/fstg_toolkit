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

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Optional, Any, Callable, Iterator, Type

import networkx as nx
import pandas as pd

from ._docker_utils import DockerHelper, DockerNotAvailableException, DockerImage

logger = logging.getLogger()


class SPMinerService:
    """Service wrapper for the SPMiner frequent subgraph pattern miner.

    Manages the Docker image lifecycle (build/load on demand) and provides a
    simple interface to run the miner on a directory of input graphs and collect
    the results.
    """

    def __init__(self):
        """Initialise the service by connecting to Docker.

        Raises
        ------
        RuntimeError
            If Docker is not available on the host system.
        """
        try:
            self.__docker_helper = DockerHelper()
        except DockerNotAvailableException as e:
            raise RuntimeError("Unable to initialize SPMiner service.") from e

        self.__docker_image: Optional[DockerImage] = None
        self.__progress_reg = re.compile(r'^\[(?P<completed>\d+)/(?P<total>\d+)]')

    def prepare(self):
        """Build or load the SPMiner Docker image if it is not already loaded.

        The image is built from the ``spminer/`` submodule located next to this
        package. Subsequent calls are no-ops if the image is already loaded.
        """
        if self.__docker_image is None:
            # TODO use an external config file?
            tag = 'spminer:latest'
            build_path = Path(__file__).parent.parent / 'spminer'
            self.__docker_image = self.__docker_helper.load_local_image(tag, build_path)

    def run(self, input_dir: Path, output_dir: Path):
        """Run the SPMiner container on a directory of graph files.

        Mounts ``input_dir`` as read-only and ``output_dir`` as read-write
        inside the container. Progress updates are yielded as they arrive.

        Parameters
        ----------
        input_dir: Path
            Directory containing the input graph files.
        output_dir: Path
            Directory where the miner will write its output.

        Yields
        ------
        tuple[int, int]
            ``(completed, total)`` progress tuples parsed from container stdout.
        """
        self.prepare()  # makes sure docker image is set

        output = self.__docker_image.run(
            volumes={str(input_dir.resolve()): {'bind': '/app/data', 'mode': 'ro'},
                     str(output_dir.resolve()): {'bind': '/app/results_batch', 'mode': 'rw'}},
            stdout=True,
            stderr=True
        )

        for line in output:
            if len(line) < 10:
                if match := self.__progress_reg.match(line):
                    yield int(match.group('completed'))-1, int(match.group('total'))
            logger.debug(line[:-1] if line[-1] == '\n' else line)


class PatternEquivalenceStrategy(ABC):
    """Abstract base class for pattern equivalence comparison strategies.

    Defines the interface for determining whether two frequent patterns are
    equivalent under different criteria (structure only, with transitions, etc.).
    """

    @classmethod
    @abstractmethod
    def equivalent(cls, p1: 'FrequentPattern', p2: 'FrequentPattern') -> bool:
        """Determine if two patterns are equivalent under this strategy.

        Parameters
        ----------
        p1 : FrequentPattern
            First pattern to compare.
        p2 : FrequentPattern
            Second pattern to compare.

        Returns
        -------
        bool
            True if patterns are equivalent, False otherwise.
        """


class PatternEquivalenceStrategyRegistry:
    """Registry for PatternEquivalenceStrategy implementations.

    Strategies self-register via the @PatternEquivalenceStrategyRegistry.register(name) decorator.
    Look up by the registered name string.
    """

    _strategies: dict[str, Type['PatternEquivalenceStrategy']] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type['PatternEquivalenceStrategy']], Type['PatternEquivalenceStrategy']]:
        """Class decorator factory that registers a strategy under the given name.

        Parameters
        ----------
        name : str
            The name key to register the strategy under.

        Returns
        -------
        Callable
            Decorator that stores the class and returns it unchanged.
        """
        def decorator(strategy_cls: Type['PatternEquivalenceStrategy']) -> Type['PatternEquivalenceStrategy']:
            cls._strategies[name] = strategy_cls
            return strategy_cls
        return decorator

    @classmethod
    def get(cls, name: str) -> Type['PatternEquivalenceStrategy']:
        """Look up a strategy class by its registered name.

        Parameters
        ----------
        name : str
            The registered name key.

        Returns
        -------
        type[PatternEquivalenceStrategy]
            The registered strategy class.

        Raises
        ------
        KeyError
            If no strategy is registered under this name.
        """
        return cls._strategies[name]

    @classmethod
    def names(cls) -> list[str]:
        """Return the names of all registered strategies.

        Returns
        -------
        list[str]
            Sorted list of registered strategy name keys.
        """
        return sorted(cls._strategies.keys())


@PatternEquivalenceStrategyRegistry.register('structure')
class PatternStructure(PatternEquivalenceStrategy):
    """Equivalence strategy based on graph structure only.

    Two patterns are equivalent if they are isomorphic as directed graphs,
    regardless of node or edge attributes.
    """

    @classmethod
    def equivalent(cls, p1: 'FrequentPattern', p2: 'FrequentPattern') -> bool:
        return nx.isomorphism.is_isomorphic(p1, p2)


@PatternEquivalenceStrategyRegistry.register('structure-transitions')
class PatternStructureTransitions(PatternEquivalenceStrategy):
    """Equivalence strategy based on structure and edge transitions.

    Two patterns are equivalent if they are isomorphic and all corresponding edges
    have the same transition attributes.
    """

    @classmethod
    def equivalent(cls, p1: 'FrequentPattern', p2: 'FrequentPattern') -> bool:
        if not nx.isomorphism.is_isomorphic(p1, p2):
            return False

        matcher = nx.isomorphism.DiGraphMatcher(p1, p2)
        if not matcher.is_isomorphic():
            return False

        return all(p1[u][v]['transition'] == p2[matcher.mapping[u]][matcher.mapping[v]]['transition']
                   for u, v in p1.edges())


@PatternEquivalenceStrategyRegistry.register('structure-regions-transitions')
class PatternStructureRegionsTransitions(PatternEquivalenceStrategy):
    """Equivalence strategy based on exact structure including regions and transitions.

    Two patterns are equivalent only if they have identical nodes and edges with all
    their attributes (regions and transitions).
    """

    @classmethod
    def equivalent(cls, p1: 'FrequentPattern', p2: 'FrequentPattern') -> bool:
        return p1.nodes(data=True) == p2.nodes(data=True) and \
            p1.edges(data=True) == p2.edges(data=True)


class FrequentPattern(nx.DiGraph):
    """A directed graph representing a frequent subgraph pattern.

    A frequent pattern is a recurring subgraph structure discovered from spatio-temporal
    graphs. It extends NetworkX's DiGraph to represent the pattern's topology, node
    attributes (e.g., brain regions), and edge attributes (e.g., temporal transitions).
    """

    def __init__(self, graph: nx.DiGraph):
        """Initialize the FrequentPattern.

        Parameters
        ----------
        graph : nx.DiGraph
            A directed graph to wrap as a frequent pattern.
        """
        super().__init__(graph)

    @staticmethod
    def from_dict(graph_dict: dict[str, Any]) -> 'FrequentPattern':
        graph = nx.DiGraph()

        for node in graph_dict['nodes']:
            graph.add_node(node['id'], **{k: v for k, v in node.items() if k != 'id'})

        for edge in graph_dict['edges']:
            graph.add_edge(edge['source'], edge['target'],
                           **{k: v for k, v in edge.items() if k not in ('source', 'target')})

        return FrequentPattern(graph)


@dataclass(frozen=True)
class FrequentPatterns:
    """A collection of frequent subgraph patterns for a single subject or group.

    Stores multiple frequent patterns discovered from a spatio-temporal graph dataset,
    each with a unique identifier. This immutable container allows iteration over the
    patterns and retrieval of their count.

    Attributes
    ----------
    patterns : dict[str, FrequentPattern]
        A mapping from pattern identifiers to FrequentPattern objects.
    """

    patterns: dict[str, FrequentPattern]

    def __len__(self) -> int:
        """Return the number of patterns in this collection.

        Returns
        -------
        int
            The count of distinct patterns.
        """
        return len(self.patterns)

    def __iter__(self) -> Iterator[tuple[str, FrequentPattern]]:
        """Iterate over pattern identifiers and objects.

        Yields
        ------
        tuple[str, FrequentPattern]
            Tuples of (pattern_id, pattern) for each pattern in the collection.
        """
        return iter(self.patterns.items())


class FrequentPatternsPopulationAnalysis:
    """Analyze frequent patterns across a population using an equivalence strategy.

    Identifies unique patterns in a multi-subject dataset and tracks which
    subjects/groups contain each unique pattern, using a specified equivalence
    criterion to group structurally similar patterns.
    """

    def __init__(self, patterns: dict[tuple[str, ...], FrequentPatterns], ids_names: tuple[str],
                 equivalence_strategy: Type[PatternEquivalenceStrategy]):
        """Initialize population analysis.

        Parameters
        ----------
        patterns : dict[tuple[str, ...], FrequentPatterns]
            Dictionary mapping subject/group ID tuples to their frequent patterns.
        ids_names : tuple[str]
            Names of the ID dimensions (e.g., ("subject", "session")).
        equivalence_strategy : Type[PatternEquivalenceStrategy]
            Strategy class to determine if two patterns are equivalent.
        """
        self.unique_patterns, self.track = self.__build_unique_patterns_track(patterns, equivalence_strategy, ids_names)

    @staticmethod
    def __build_unique_patterns_track(patterns: dict[tuple[str, ...], FrequentPatterns],
                                      equivalence_strategy: Type[PatternEquivalenceStrategy],
                                      ids_names: tuple[str]) -> tuple[list[FrequentPattern], pd.DataFrame]:
        """Build unique patterns list and tracking table.

        Identifies unique patterns across the population, grouping equivalent
        patterns together and creating a mapping of which subjects contain each.

        Parameters
        ----------
        patterns : dict[tuple[str, ...], FrequentPatterns]
            Dictionary mapping subject/group ID tuples to their frequent patterns.
        equivalence_strategy : Type[PatternEquivalenceStrategy]
            Strategy class for determining pattern equivalence.
        ids_names : tuple[str]
            Names of the ID dimensions.

        Returns
        -------
        tuple[list[FrequentPattern], pd.DataFrame]
            A tuple containing:
            - List of unique patterns (each equivalence class represented once)
            - DataFrame tracking which subjects contain which unique patterns,
              indexed by ID columns with an 'idx' column for unique pattern index.
        """
        unique = []
        track_records = []

        for ids, patterns in patterns.items():
            for name, pattern in patterns:
                idx = next((i for i, p in enumerate(unique)
                            if equivalence_strategy.equivalent(pattern, p)), None)

                if idx is None:
                    unique.append(pattern)
                    idx = len(unique) - 1

                track_records.append(dict(zip(ids_names, ids)) | {'idx': idx})

        return unique, pd.DataFrame.from_records(track_records).set_index(list(ids_names))

    def get_counts(self, factors: list[str]) -> pd.DataFrame:
        """Count occurrences of each unique pattern, optionally grouped by factors.

        Aggregates the tracking data to compute how many subjects/groups contain each
        unique pattern. If factors are specified, counts are computed separately for
        each combination of factor values.

        Parameters
        ----------
        factors : list[str]
            Column names from the tracking DataFrame to group by (e.g., ['session']).
            Pass an empty list to get counts across all subjects.

        Returns
        -------
        pd.DataFrame
            A DataFrame with unique pattern indices as rows and 'Count' column containing
            the number of subjects with each pattern. If factors are provided, the result
            is multi-indexed by the factor columns and pattern index 'idx'.
        """
        if factors:
            result = pd.concat({group: data.reset_index('Subject').groupby('idx').count().rename(columns={'Subject': 'Count'})
                                for group, data in self.track.groupby(factors)},
                               axis=0)
            result.index.names = [*factors, 'idx']
            return result
        else:
            return self.track.reset_index('Subject').groupby('idx').count().rename(columns={'Subject': 'Count'})

    @staticmethod
    def _iter_counts(counts: pd.DataFrame, factors: list[str]) -> Iterator[tuple[dict[str, Any], int, int]]:
        """Iterate over counts rows with unpacked factor dict, idx, and count.

        Handles the conditional unpacking of rows based on whether factors are present,
        yielding a consistent (factor_dict, idx, count) tuple for each row.

        Parameters
        ----------
        counts : pd.DataFrame
            DataFrame from get_counts() with index levels for factors (if any) and 'idx'.
        factors : list[str]
            Column names that were used to group in get_counts().

        Yields
        ------
        tuple[dict[str, Any], int, int]
            (factor_dict, idx, count) where factor_dict is empty if factors is empty.
        """
        if factors:
            for (*factor_vals, idx), count in counts.itertuples(name=None):
                yield dict(zip(factors, factor_vals)), idx, count
        else:
            for idx, count in counts.itertuples(name=None):
                yield {}, idx, count

    def get_patterns_per_region(self, factors: list[str]) -> pd.DataFrame:
        """Count pattern occurrences per brain region, optionally grouped by factors.

        For each unique pattern, extracts all regions present in its nodes. Each
        region occurrence is weighted by the number of subjects that have the pattern
        in the given factor group.

        Parameters
        ----------
        factors : list[str]
            Column names from the tracking DataFrame to group by.
            Pass an empty list to get counts across all subjects.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns ``[Region, Count, PatternIndices, *factors]``,
            where ``PatternIndices`` is a sorted comma-separated string of 1-based
            pattern indices that contain at least one node in that region.
        """
        counts = self.get_counts(factors)
        records: list[dict[str, Any]] = []

        for factor_dict, idx, count in self._iter_counts(counts, factors):
            pattern = self.unique_patterns[idx]
            for _, node_data in pattern.nodes(data=True):
                records.append({
                    'Region': node_data['region'],
                    'Count': count,
                    'PatternIdx': idx + 1,
                    **factor_dict,
                })

        df = pd.DataFrame.from_records(records)
        group_cols = ['Region'] + factors
        return df.groupby(group_cols, as_index=False).agg(
            Count=('Count', 'sum'),
            PatternIndices=('PatternIdx', lambda x: ', '.join(str(i) for i in sorted(set(x))))
        )

    def get_temporal_dynamics(self, factors: list[str]) -> pd.DataFrame:
        """Extract temporal edge dynamics per region, optionally grouped by factors.

        For each unique pattern, extracts temporal edges (those with a ``transition``
        attribute) and records the source node's region and the transition type.
        Counts are weighted by the number of subjects that have the pattern in each
        factor group.

        Parameters
        ----------
        factors : list[str]
            Column names from the tracking DataFrame to group by.
            Pass an empty list to get counts across all subjects.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns ``[Region, Transition, Count, PatternIndices, *factors]``,
            where ``PatternIndices`` is a sorted comma-separated string of 1-based
            pattern indices that have at least one temporal edge matching that region
            and transition type.
        """
        counts = self.get_counts(factors)
        records: list[dict[str, Any]] = []

        for factor_dict, idx, count in self._iter_counts(counts, factors):
            pattern = self.unique_patterns[idx]
            for u, v, edge_data in pattern.edges(data=True):
                if 'transition' in edge_data:
                    records.append({
                        'Region': pattern.nodes[u]['region'],
                        'Transition': str(edge_data['transition']),
                        'Count': count,
                        'PatternIdx': idx + 1,
                        **factor_dict,
                    })

        df = pd.DataFrame.from_records(records)
        group_cols = ['Region', 'Transition'] + factors
        return df.groupby(group_cols, as_index=False).agg(
            Count=('Count', 'sum'),
            PatternIndices=('PatternIdx', lambda x: ', '.join(str(i) for i in sorted(set(x))))
        )

    @staticmethod
    def _collect_all_regions(unique_patterns: list['FrequentPattern']) -> tuple[list[str], dict[str, int]]:
        """Collect and index all regions across unique patterns.

        Parameters
        ----------
        unique_patterns : list[FrequentPattern]
            Patterns to scan.

        Returns
        -------
        tuple[list[str], dict[str, int]]
            Sorted region labels and region-to-matrix-index mapping.
        """
        all_regions: set[str] = set()
        for pattern in unique_patterns:
            for _, node_data in pattern.nodes(data=True):
                all_regions.add(node_data['region'])
        region_labels = sorted(all_regions)
        return region_labels, {r: i for i, r in enumerate(region_labels)}

    @staticmethod
    def _count_spatial_edge_pairs(pattern: 'FrequentPattern', count: int) -> dict[tuple[str, str], int]:
        """Return weighted spatial-edge region pairs for a single pattern.

        Only considers edges without a ``transition`` attribute that connect
        two distinct regions.

        Parameters
        ----------
        pattern : FrequentPattern
            Pattern whose spatial edges are scanned.
        count : int
            Weight applied to each pair found.

        Returns
        -------
        dict[tuple[str, str], int]
            Mapping from sorted ``(r1, r2)`` region pairs to weighted counts.
        """
        pairs: dict[tuple[str, str], int] = {}
        for u, v, edge_data in pattern.edges(data=True):
            if 'transition' not in edge_data:
                r1: str = pattern.nodes[u]['region']
                r2: str = pattern.nodes[v]['region']
                if r1 != r2:
                    pair = (r1, r2) if r1 <= r2 else (r2, r1)
                    pairs[pair] = pairs.get(pair, 0) + count
        return pairs

    @staticmethod
    def _build_symmetric_matrix(pairs: dict[tuple[str, str], int],
                                 region_idx: dict[str, int], n: int) -> list[list[int]]:
        """Build a symmetric co-occurrence matrix from region pairs.

        Parameters
        ----------
        pairs : dict[tuple[str, str], int]
            Sorted region pairs and their counts.
        region_idx : dict[str, int]
            Region-to-matrix-index mapping.
        n : int
            Matrix dimension.

        Returns
        -------
        list[list[int]]
            ``n×n`` symmetric matrix with pair counts.
        """
        matrix = [[0] * n for _ in range(n)]
        for (r1, r2), count in pairs.items():
            i, j = region_idx[r1], region_idx[r2]
            matrix[i][j] += count
            matrix[j][i] += count
        return matrix

    def get_region_co_occurrence(self, factors: list[str]) -> dict[tuple[str, ...], tuple[list[str], list[list[int]]]]:
        """Compute region co-occurrence matrices from spatial edges, optionally grouped by factors.

        For each spatial edge (no ``transition`` attribute, connecting different regions),
        records the sorted region pair. Counts are weighted by the number of subjects
        that have the pattern in each factor group.

        Parameters
        ----------
        factors : list[str]
            Column names from the tracking DataFrame to group by.
            Pass an empty list to get a single co-occurrence matrix.

        Returns
        -------
        dict[tuple[str, ...], tuple[list[str], list[list[int]]]]
            A dictionary mapping factor-group tuples (or ``('',)`` if no factors) to
            a tuple of ``(region_labels_sorted, symmetric_2d_list)`` where the 2D list
            contains co-occurrence counts between regions.
        """
        counts = self.get_counts(factors)
        region_labels, region_idx = self._collect_all_regions(self.unique_patterns)
        n = len(region_labels)

        group_pairs: dict[tuple[str, ...], dict[tuple[str, str], int]] = {}
        for factor_dict, idx, count in self._iter_counts(counts, factors):
            key = tuple(factor_dict[f] for f in factors) if factors else ()
            group_pairs.setdefault(key, {})
            for pair, c in self._count_spatial_edge_pairs(self.unique_patterns[idx], count).items():
                group_pairs[key][pair] = group_pairs[key].get(pair, 0) + c

        return {key: (region_labels, self._build_symmetric_matrix(pairs, region_idx, n))
                for key, pairs in group_pairs.items()}

    @staticmethod
    def _get_group_by_level_param(data: pd.DataFrame,
                                  exclude_factors: list[str]) -> 'str | list[str]':
        """Compute the groupby level parameter after excluding factor levels.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame whose index names are inspected.
        exclude_factors : list[str]
            Index level names to exclude.

        Returns
        -------
        str | list[str]
            Single level name if only one remains, else a list.
        """
        remaining = [lvl for lvl in data.index.names if lvl not in exclude_factors]
        return remaining[0] if len(remaining) == 1 else remaining

    @staticmethod
    def _increment_pattern_matrix(matrix: list[list[int]], indices: list[int]) -> None:
        """Increment co-occurrence counts for a subject's pattern indices.

        Parameters
        ----------
        matrix : list[list[int]]
            Co-occurrence matrix updated in-place.
        indices : list[int]
            Pattern indices present for one subject.
        """
        for i_val in indices:
            matrix[i_val][i_val] += 1
        for i_val, j_val in combinations(indices, 2):
            matrix[i_val][j_val] += 1
            matrix[j_val][i_val] += 1

    def _build_pattern_co_occurrence_matrix(self, track_data: pd.DataFrame,
                                            level_param: 'str | list[str]',
                                            n: int) -> list[list[int]]:
        """Build a pattern co-occurrence matrix for a single factor group.

        Parameters
        ----------
        track_data : pd.DataFrame
            Subset of the tracking DataFrame for one group.
        level_param : str | list[str]
            Level parameter passed to ``groupby``.
        n : int
            Number of unique patterns (matrix dimension).

        Returns
        -------
        list[list[int]]
            ``n×n`` symmetric co-occurrence matrix.
        """
        matrix = [[0] * n for _ in range(n)]
        for _, subject_data in track_data.groupby(level=level_param):
            self._increment_pattern_matrix(matrix, list(subject_data['idx'].unique()))
        return matrix

    def get_pattern_co_occurrence(self, factors: list[str]) -> dict[tuple[str, ...], list[list[int]]]:
        """Compute pattern co-occurrence matrices, optionally grouped by factors.

        For each subject in a factor group, finds all pattern indices the subject has,
        then increments the co-occurrence counter for every pair of patterns.

        Parameters
        ----------
        factors : list[str]
            Column names from the tracking DataFrame to group by.
            Pass an empty list to get a single co-occurrence matrix.

        Returns
        -------
        dict[tuple[str, ...], list[list[int]]]
            A dictionary mapping factor-group tuples (or ``('',)`` if no factors) to
            a symmetric 2D list of size ``len(unique_patterns)``, where cell ``(i, j)``
            is the number of subjects that have both pattern *i* and pattern *j*.
        """
        n = len(self.unique_patterns)
        result: dict[tuple[str, ...], list[list[int]]] = {}

        if factors:
            for group, group_data in self.track.groupby(factors):
                key = group if isinstance(group, tuple) else (group,)
                level_param = self._get_group_by_level_param(group_data, factors)
                result[key] = self._build_pattern_co_occurrence_matrix(group_data, level_param, n)
        else:
            level_param = self._get_group_by_level_param(self.track, [])
            result[()] = self._build_pattern_co_occurrence_matrix(self.track, level_param, n)

        return result

    def get_occurrence_histogram(self, factors: list[str]) -> pd.DataFrame:
        """Build a histogram of pattern occurrence counts, optionally grouped by factors.

        Computes how many patterns share the same occurrence count. For example, if 5
        patterns each appear in exactly 3 subjects, the histogram will have a row with
        ``Occurrences=3, Patterns=5``.

        Parameters
        ----------
        factors : list[str]
            Column names from the tracking DataFrame to group by.
            Pass an empty list to get a single histogram.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns ``[Occurrences, Patterns, PatternIndices, *factors]``,
            where ``PatternIndices`` is a sorted comma-separated string of 1-based
            pattern indices that have that occurrence count.
        """
        counts = self.get_counts(factors)
        records: list[dict[str, Any]] = []

        for factor_dict, idx, count in self._iter_counts(counts, factors):
            records.append({
                'Occurrences': count,
                'PatternIdx': idx + 1,
                **factor_dict,
            })

        df = pd.DataFrame.from_records(records)
        group_cols = ['Occurrences'] + factors
        return df.groupby(group_cols, as_index=False).agg(
            Patterns=('PatternIdx', 'count'),
            PatternIndices=('PatternIdx', lambda x: ', '.join(str(i) for i in sorted(x)))
        )

    def get_pattern_complexity(self, factors: list[str]) -> pd.DataFrame:
        """Compute pattern complexity (node count) distribution, optionally grouped by factors.

        For each unique pattern, computes its size as the number of nodes. The size is
        weighted by the number of subjects that have the pattern in each factor group.

        Parameters
        ----------
        factors : list[str]
            Column names from the tracking DataFrame to group by.
            Pass an empty list to get counts across all subjects.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns ``[Size, Count, PatternIndices, *factors]``,
            where ``PatternIndices`` is a sorted comma-separated string of 1-based
            pattern indices that have that node count.
        """
        counts = self.get_counts(factors)
        records: list[dict[str, Any]] = []

        for factor_dict, idx, count in self._iter_counts(counts, factors):
            pattern = self.unique_patterns[idx]
            records.append({
                'Size': len(pattern.nodes()),
                'Count': count,
                'PatternIdx': idx + 1,
                **factor_dict,
            })

        df = pd.DataFrame.from_records(records)
        group_cols = ['Size'] + factors
        return df.groupby(group_cols, as_index=False).agg(
            Count=('Count', 'sum'),
            PatternIndices=('PatternIdx', lambda x: ', '.join(str(i) for i in sorted(x)))
        )
