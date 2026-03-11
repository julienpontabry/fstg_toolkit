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
from pathlib import Path
from typing import Optional, Any, Iterator, Type

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
    def name(cls) -> str:
        """Return the name identifier of this equivalence strategy.

        Returns
        -------
        str
            A descriptive name for the strategy (e.g., "structure", "structure-transitions").
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

class PatternStructure(PatternEquivalenceStrategy):
    """Equivalence strategy based on graph structure only.

    Two patterns are equivalent if they are isomorphic as directed graphs,
    regardless of node or edge attributes.
    """

    @classmethod
    def name(cls) -> str:
        return "structure"

    @classmethod
    def equivalent(cls, p1: 'FrequentPattern', p2: 'FrequentPattern') -> bool:
        return nx.isomorphism.is_isomorphic(p1, p2)


class PatternStructureTransitions(PatternEquivalenceStrategy):
    """Equivalence strategy based on structure and edge transitions.

    Two patterns are equivalent if they are isomorphic and all corresponding edges
    have the same transition attributes.
    """

    @classmethod
    def name(cls) -> str:
        return "structure-transitions"

    @classmethod
    def equivalent(cls, p1: 'FrequentPattern', p2: 'FrequentPattern') -> bool:
        if not nx.isomorphism.is_isomorphic(p1, p2):
            return False

        matcher = nx.isomorphism.DiGraphMatcher(p1, p2)
        if not matcher.is_isomorphic():
            return False

        return all(p1[u][v]['transition'] == p2[matcher.mapping[u]][matcher.mapping[v]]['transition']
                   for u, v in p1.edges())


class PatternStructureRegionsTransitions(PatternEquivalenceStrategy):
    """Equivalence strategy based on exact structure including regions and transitions.

    Two patterns are equivalent only if they have identical nodes and edges with all
    their attributes (regions and transitions).
    """

    @classmethod
    def name(cls) -> str:
        return "structure-regions-transitions"

    @classmethod
    def equivalent(cls, p1: 'FrequentPattern', p2: 'FrequentPattern') -> bool:
        return p1.nodes(data=True) == p2.nodes(data=True) and \
            p1.edges(data=True) == p2.edges(data=True)


class FrequentPattern(nx.DiGraph):
    def __init__(self, graph: nx.DiGraph):
        """Initialize the FrequentPattern."""
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
    patterns: dict[str, FrequentPattern]

    def __len__(self) -> int:
        return len(self.patterns)

    def __iter__(self) -> Iterator[tuple[str, FrequentPattern]]:
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
