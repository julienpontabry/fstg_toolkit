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

"""Defines helpers for inputs/outputs."""

import datetime
import json
import logging
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, IO, Protocol, Optional, Generator, Iterable
from zipfile import ZipFile, ZipInfo

import networkx as nx
import numpy as np
import pandas as pd

from .frequent import FrequentPatterns, FrequentPattern
from .graph import SpatioTemporalGraph, RC5
from .utils import split_factors_from_name

logger = logging.getLogger()


class _SpatioTemporalGraphEncoder(json.JSONEncoder):
    """JSON encoder for spatio-temporal graph.

    The sets are converted to lists and the RC5 objects are converted to their
    names as strings. The rest is left untouched.
    """

    def default(self, obj):
        """Serialize RC5 enum values and sets to JSON-compatible types.

        Parameters
        ----------
        obj: Any
            The object to serialize.

        Returns
        -------
        Any
            The JSON-serializable representation: the RC5 name string for
            :class:`RC5` instances, a list for sets, or the default encoding
            for all other types.
        """
        if isinstance(obj, RC5):
            return obj.name
        elif isinstance(obj, set):
            return list(obj)
        else:
            return super().default(obj)


def _spatio_temporal_object_hook(obj: dict) -> dict:
    """Object hook for decoding JSON-encoded spatio-temporal graph.

    The values of 'areas' fields are converted from list to set (it is about areas id
    in the network). The strings describing RC5 transitions in temporal edges are used
    to build in place the actual RC5 transition. The rest is left untouched.

    Parameters
    ----------
    obj: dict
        A JSON object to decode.

    Returns
    -------
    dict
        The decoded JSON object.
    """
    if 'areas' in obj:
        obj['areas'] = set(obj['areas'])
    elif 'type' in obj and obj['type'] == 'temporal':
        obj['transition'] = RC5.from_name(obj['transition'])

    return obj


class DataHandler(Protocol):
    """Protocol defining the interface for data format handlers.

    A ``DataHandler`` is responsible for serializing and deserializing a
    specific type of data to and from file-like objects, and for mapping
    between human-readable names and their on-disk filenames.
    """

    def matches(self, filename: str) -> bool:
        """Check if the handler can handle a file from its filename.

        Parameters
        ----------
        filename : str
            The filename to test.

        Returns
        -------
        bool
            ``True`` if this handler can process the given filename.
        """

    def serialize(self, item: Any, fp: IO, **context: Any) -> None:
        """Serialize the item to a file-like object.

        Parameters
        ----------
        item : Any
            The data object to serialize.
        fp : IO
            Writable binary file-like object.
        **context : Any
            Optional extra keyword arguments passed through to the handler.
        """

    def deserialize(self, fp: IO, **context: Any) -> Any:
        """Deserialize the item from a file-like object.

        Parameters
        ----------
        fp : IO
            Readable binary file-like object.
        **context : Any
            Optional extra keyword arguments required by the handler.

        Returns
        -------
        Any
            The deserialized data object.
        """

    def filename2name(self, filename: str) -> str:
        """Convert a filename to its corresponding name.

        Parameters
        ----------
        filename : str
            On-disk filename (e.g. ``"metrics_foo.csv"``).

        Returns
        -------
        str
            Human-readable name (e.g. ``"foo"``).
        """

    def name2filename(self, name: str) -> str:
        """Convert a name to its corresponding filename.

        Parameters
        ----------
        name : str
            Human-readable name.

        Returns
        -------
        str
            On-disk filename.
        """


class NoDataHandlerFound(TypeError):
    """Raised when no registered handler matches a given filename or name."""

    def __init__(self, name: str) -> None:
        super().__init__(f"No handler found for \"{name}\".")


class DataRegistry:
    """Central registry mapping data kinds to their :class:`DataHandler` instances.

    Handlers are registered with :meth:`register` and looked up dynamically
    from filenames via :meth:`resolve`.
    """

    _handlers: dict[str, DataHandler] = {}

    @classmethod
    def register(cls, kind: str):
        """Class decorator that registers a handler under the given kind key.

        Parameters
        ----------
        kind : str
            The logical kind label (e.g. ``"graphs"``, ``"metrics"``).

        Returns
        -------
        Callable
            A decorator that stores the decorated class in the registry.

        Examples
        --------
        >>> @DataRegistry.register('my_kind')
        ... class MyHandler: ...
        """
        def decorator(handler_cls):
            logger.debug(f"Registering handler '{handler_cls.__name__}' for kind '{kind}'.")
            cls._handlers[kind] = handler_cls
            return handler_cls
        return decorator

    @classmethod
    def resolve(cls, filename: str) -> Optional[tuple[str, DataHandler]]:
        """Find the handler matching the given filename.

        Parameters
        ----------
        filename : str
            The filename to match against all registered handlers.

        Returns
        -------
        tuple[str, DataHandler] or None
            A ``(kind, handler)`` pair if a match is found, otherwise ``None``.
        """
        for kind, handler in cls._handlers.items():
            if handler.matches(filename):
                logger.debug(f"Resolved filename '{filename}' to handler kind '{kind}'.")
                return kind, handler
        logger.debug(f"No handler found for filename '{filename}'.")
        return None

    @classmethod
    def classify(cls, filenames: list[str]) -> dict[str, list[str]]:
        """Group filenames by their resolved handler kind.

        Parameters
        ----------
        filenames : list[str]
            List of filenames to classify.

        Returns
        -------
        dict[str, list[str]]
            A mapping from kind label to the list of matching filenames.
            Filenames that do not match any handler are silently ignored.
        """
        results = {}
        for filename in filenames:
            if resolved := cls.resolve(filename):
                kind, _ = resolved
                results.setdefault(kind, []).append(filename)
        logger.debug(f"Classified {len(filenames)} filename(s) into {len(results)} kind(s): "
                     f"{', '.join(f'{k}({len(v)})' for k, v in results.items()) or 'none'}.")
        return results

    @classmethod
    def filename2name(cls, filename: str) -> str:
        """Convert a filename to its corresponding name.

        Parameters
        ----------
        filename : str
            The on-disk filename of the data item.

        Returns
        -------
        str:
            The corresponding human-readable name of the data item.

        Raises
        ------
        NoDataHandlerFound
            If no handler is registered for ``filename``.
        """
        if resolved := cls.resolve(filename):
            _, handler = resolved
            return handler.filename2name(filename)
        else:
            raise NoDataHandlerFound(filename)

    @classmethod
    def name2filename(cls, name: str, kind: str) -> str:
        """Convert a logical name to its on-disk filename for a given kind.

        Parameters
        ----------
        name : str
            Human-readable name of the data item.
        kind : str
            The handler kind under which the name should be resolved.

        Returns
        -------
        str
            The corresponding on-disk filename.

        Raises
        ------
        NoDataHandlerFound
            If no handler is registered for ``kind``.
        """
        if handler := cls._handlers.get(kind, None):
            return handler.name2filename(name)
        else:
            raise NoDataHandlerFound(name)

    @classmethod
    def serialize(cls, filename: str, item: Any, fp: IO, **context) -> None:
        """Serialize ``item`` to ``fp`` using the handler matched by ``filename``.

        Parameters
        ----------
        filename : str
            Used to look up the appropriate handler.
        item : Any
            Data object to serialize.
        fp : IO
            Writable binary file-like object.
        **context : Any
            Extra keyword arguments forwarded to the handler.

        Raises
        ------
        NoDataHandlerFound
            If no handler matches ``filename``.
        """
        if resolved := cls.resolve(filename):
            logger.debug(f"Serializing '{filename}'.")
            _, handler = resolved
            handler.serialize(item, fp, **context)
        else:
            raise NoDataHandlerFound(filename)

    @classmethod
    def deserialize(cls, filename: str, fp: IO, **context: Any) -> tuple[str, Any]:
        """Deserialize an item from ``fp`` using the handler matched by ``filename``.

        Parameters
        ----------
        filename : str
            Used to look up the appropriate handler and derive the item name.
        fp : IO
            Readable binary file-like object.
        **context : Any
            Extra keyword arguments forwarded to the handler.

        Returns
        -------
        tuple[str, Any]
            A ``(name, item)`` pair where *name* is the human-readable identifier
            derived from ``filename``.

        Raises
        ------
        NoDataHandlerFound
            If no handler matches ``filename``.
        """
        if resolved := cls.resolve(filename):
            logger.debug(f"Deserializing '{filename}'.")
            _, handler = resolved
            item = handler.deserialize(fp, **context)
            name = handler.filename2name(filename)
            return name, item
        else:
            raise NoDataHandlerFound(filename)


@DataRegistry.register('areas')
class AreasDescHandler(Protocol):
    """Handler for the areas descriptor CSV file (``areas.csv``)."""

    filename: str = 'areas.csv'

    @staticmethod
    def matches(filename: str) -> bool:
        return filename.lower() == AreasDescHandler.filename

    @staticmethod
    def serialize(item: pd.DataFrame, fp: IO) -> None:
        item.to_csv(fp)

    @staticmethod
    def deserialize(fp: IO) -> pd.DataFrame:
        return pd.read_csv(fp, index_col='Id_Area')

    @staticmethod
    def filename2name(_) -> str:
        return Path(AreasDescHandler.filename).stem

    @staticmethod
    def name2filename(_) -> str:
        return AreasDescHandler.filename


@DataRegistry.register('graphs')
class GraphHandler:
    """Handler for :class:`~fstg_toolkit.SpatioTemporalGraph` stored as JSON files.

    Filenames must end with ``.json`` and must *not* match the
    ``motifs_enriched_*.json`` pattern (which is reserved for motif data).
    """

    pattern: re.Pattern = re.compile(r'^(?!.*motifs_enriched_.+\.json$)(?P<name>.*)\.json$')

    @staticmethod
    def matches(filename: str) -> bool:
        return GraphHandler.pattern.match(filename) is not None

    @staticmethod
    def serialize(item: SpatioTemporalGraph, fp: IO) -> None:
        graph_dict = nx.json_graph.node_link_data(item, edges='edges')
        graph_json = json.dumps(graph_dict, cls=_SpatioTemporalGraphEncoder)
        fp.write(graph_json.encode('utf-8'))

    @staticmethod
    def deserialize(fp: IO, **context: Any) -> SpatioTemporalGraph:
        areas = context.get('areas')
        if areas is None:
            raise ValueError("Graph deserialization requires 'areas' in context.")

        graph_dict = json.load(fp, object_hook=_spatio_temporal_object_hook)
        graph = nx.json_graph.node_link_graph(graph_dict, edges='edges')
        return SpatioTemporalGraph(graph, areas)

    @staticmethod
    def filename2name(filename: str) -> str:
        if match := GraphHandler.pattern.match(filename):
            return match.group('name')
        else:
            return 'graph'

    @staticmethod
    def name2filename(name: str) -> str:
        return f'{name}.json'


@DataRegistry.register('matrices')
class MatrixHandler(Protocol):
    """Handler for correlation matrices stored as ``.npy`` files."""

    pattern: re.Pattern = re.compile(r'.+\.npy$')

    @staticmethod
    def matches(filename: str) -> bool:
        return MatrixHandler.pattern.match(filename) is not None

    @staticmethod
    def serialize(item: np.ndarray, fp: IO) -> None:
        np.save(fp, item)

    @staticmethod
    def deserialize(fp: IO) -> np.ndarray:
        return np.load(fp, allow_pickle=False)

    @staticmethod
    def filename2name(filename: str) -> str:
        return Path(filename).stem

    @staticmethod
    def name2filename(name: str) -> str:
        return f'{name}.npy'


@DataRegistry.register('metrics')
class MetricsHandler:
    """Handler for metric data frames stored as ``metrics_<name>.csv`` files."""

    pattern: re.Pattern = re.compile(r'^metrics_(?P<name>.+)\.csv$')

    @staticmethod
    def matches(filename: str) -> bool:
        return MetricsHandler.pattern.match(filename) is not None

    @staticmethod
    def serialize(item: pd.DataFrame, fp: IO) -> None:
        df: pd.DataFrame = item.copy()

        # serialize dictionaries if any
        for col in df.select_dtypes(include=['object']).columns:
            if any(isinstance(x, dict) and any(isinstance(k, RC5) for k in x.keys()) for x in df[col].dropna()):
                logger.debug(f"Serializing RC5 dict column '{col}'.")
                df[col] = df[col].apply(
                    lambda x: json.dumps({k.name: v for k, v in x.items()}) if isinstance(x, dict) else x)

        # serialize multi-index if any
        # NOTE not compatible with floating points index elements
        if isinstance(df.index, pd.MultiIndex):
            logger.debug("Flattening MultiIndex to dot-separated strings.")
            df.index = pd.Index(['.'.join([str(e) for e in idx]) for idx in df.index],
                                name='.'.join(df.index.names))

        # serialize multi-columns if any
        if isinstance(df.columns, pd.MultiIndex):
            logger.debug("Flattening MultiIndex columns to dot-separated strings.")
            df.columns = ['.'.join(c) for c in df.columns]

        df.to_csv(fp)

    @staticmethod
    def __to_rc5_if_possible(d: dict[str, Any]) -> dict[str | RC5, Any]:
        """Convert string keys that match RC5 transition names to actual RC5 enum members.

        Parameters
        ----------
        d: dict[str, Any]
            A dictionary whose keys may be RC5 transition name strings.

        Returns
        -------
        dict[RC5, Any]
            A new dictionary with RC5-recognizable keys replaced by the
            corresponding :class:`RC5` enum members. Keys that do not match any
            RC5 transition are dropped.
        """
        return {RC5.from_name(k): v for k, v in d.items() if RC5.includes(k)}

    @staticmethod
    def deserialize(fp: IO) -> pd.DataFrame:
        df = pd.read_csv(fp, index_col=0)

        # deserialize dictionaries if any
        for col in df.select_dtypes(include=['object']).columns:
            if any(isinstance(x, str) and x.strip().startswith('{') and x.strip().endswith('}')\
                   for x in df[col].dropna()):
                logger.debug(f"Deserializing RC5 dict column '{col}'.")
                df[col] = df[col].apply(lambda x: MetricsHandler.__to_rc5_if_possible(json.loads(x))
                if isinstance(x, str) and x.strip().startswith('{') and x.strip().endswith('}') else x)

        # deserialize multi-index if any
        if '.' in df.index.name and all('.' in idx for idx in df.index):
            logger.debug("Restoring MultiIndex from dot-separated index strings.")
            tuples = [tuple(int(i) if i.isdigit() else i
                             for i in idx.split('.'))
                      for idx in df.index]
            df.index = pd.MultiIndex.from_tuples(tuples, names=df.index.name.split('.'))

        # deserialize multi-columns if any
        if all('.' in c for c in df.columns):
            logger.debug("Restoring MultiIndex columns from dot-separated column strings.")
            df.columns = pd.MultiIndex.from_tuples([tuple(c.split('.')) for c in df.columns])

        return df

    @staticmethod
    def filename2name(filename: str) -> str:
        if match := MetricsHandler.pattern.match(filename):
            return match.group('name')
        else:
            return 'metrics'

    @staticmethod
    def name2filename(name: str) -> str:
        return f'metrics_{name}.csv'


@DataRegistry.register('frequent_patterns')
class FrequentPatternsHandler:
    """Handler for frequent pattern from SPMiner

    The filenames are ``<subject>/motifs_enriched_<mode>.json``, where
    mode is ``s``, ``t`` or ``st``.
    """

    pattern: re.Pattern = re.compile(r'^(?P<subject>.+)/motifs_enriched_(?P<mode>s|t|st)\.json$')

    @staticmethod
    def matches(filename: str) -> bool:
        return FrequentPatternsHandler.pattern.match(filename) is not None

    @staticmethod
    def serialize(item: FrequentPatterns, fp: IO) -> None:
        patterns_dict = {name: nx.json_graph.node_link_data(pattern, edges='edges') for name, pattern in item}
        patterns_json = json.dumps(patterns_dict, cls=_SpatioTemporalGraphEncoder)
        fp.write(patterns_json.encode('utf-8'))

    @staticmethod
    def deserialize(fp: IO) -> FrequentPatterns:
        patterns_dict = json.load(fp, object_hook=_spatio_temporal_object_hook)
        patterns = {name: nx.json_graph.node_link_graph(pattern_dict, edges='edges')
                    for name, pattern_dict in patterns_dict.items()}
        return FrequentPatterns(patterns)

    @staticmethod
    def filename2name(filename: str) -> str:
        if match := FrequentPatternsHandler.pattern.match(filename):
            return ','.join(match.groups())
        return filename

    @staticmethod
    def name2filename(name: str) -> str:
        subject, mode = name.split(',')
        return f'{subject}/motifs_enriched_{mode}.json'


@dataclass(frozen=True)
class DataLoader:
    """Read-only accessor for a ZIP archive produced by :class:`DataSaver`.

    On construction the archive is opened once to build an inventory of all
    known data files, grouped by handler kind.

    Methods are provided to load (lazily or not) the elements of the dataset,
    such as the correlation matrices, the graphs, the metrics, etc.

    Parameters
    ----------
    filepath : Path
        Path to an existing ZIP archive.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist or points to a directory.
    """

    filepath: Path
    _inventory: dict[str, list[str]] = field(default_factory=lambda: {})

    def __post_init__(self):
        """Validate that the provided filepath points to an existing file.

        Raises
        ------
        FileNotFoundError
            If the path does not exist or is a directory.
        """
        if not self.filepath.exists() or self.filepath.is_dir():
            raise FileNotFoundError()

        logger.info(f"Opening archive '{self.filepath}' and building inventory.")
        with self.__open() as zfp:
            self._inventory.update(DataRegistry.classify(zfp.namelist()))
        logger.debug(f"Inventory built: {self._inventory}.")

    def __str__(self) -> str:
        inventory = [f"{kind}: {len(filenames)} filename(s) {"(ex: " + filenames[0] + ")" if filenames else ""}"
                     for kind, filenames in self._inventory.items()]
        return f"DataLoader(filepath={self.filepath}, inventory=[{', '.join(inventory)}])"

    @contextmanager
    def __open(self) -> Generator[ZipFile, None, None]:
        """Open the underlying ZIP archive as a context manager.

        Yields
        ------
        ZipFile
            An open :class:`~zipfile.ZipFile` object.
        """
        with ZipFile(self.filepath) as zfp:
            yield zfp

    @staticmethod
    def __load(zfp: ZipFile, filename: str, **context: Any) -> Optional[Any]:
        """Load and deserialize a single file from an open ZIP archive.

        Parameters
        ----------
        zfp : ZipFile
            Open ZIP archive.
        filename : str
            Name of the entry to load.
        **context : Any
            Extra keyword arguments forwarded to the handler's ``deserialize`` method.

        Returns
        -------
        tuple[str, Any] or None
            ``(name, item)`` on success, ``None`` if no handler is found.
        """
        logger.debug(f"Loading '{filename}' from archive.")
        with zfp.open(filename) as fp:
            try:
                return DataRegistry.deserialize(filename, fp, **context)
            except NoDataHandlerFound as e:
                logger.error(f"Unable to load item \"{filename}\": {e}.")

    def extract_files(self, output: Path, kinds: str|list[str]) -> tuple[int, Generator[str, None, None]]:
        """Extract the files of the given kinds to the given output directory.

        This method returns a generator yield, so it is required to iterate on so
        the extract can happen.

        Parameters
        ----------
        output : Path
            The directory to extract files to.
        kinds: str|list[str]
            The kinds of files to extract.

        Returns
        ------
        tuple[int, Generator[str, None, None]]
            The total number of files and a generator to the extraction that yields
            the name of the file that has been extracted.

        Raises
        ------
        ValueError: if output path does not exist or is not a directory.
        """
        if not output.exists() or not output.is_dir():
            raise ValueError(f"Output path '{output}' does not exist or is not a directory.")

        if isinstance(kinds, str):
            kinds = [kinds]

        logger.debug(f"Extracting files of kinds {kinds} to '{output}'.")

        total = sum([len(self._inventory.get(kind, [])) for kind in kinds])
        logger.debug(f"Total number of files to extract: {total}.")

        def gen():
            # TODO parallelize the graph extraction?
            with self.__open() as zfp:
                for kind in kinds:
                    for filename in self._inventory.get(kind, []):
                        zfp.extract(filename, output)
                        yield filename

        return total, gen()

    def load_areas(self) -> Optional[pd.DataFrame]:
        """Load the areas descriptor data frame from the archive.

        Returns
        -------
        pd.DataFrame or None
            The areas data frame, or ``None`` if no areas file is present.
        """
        if filenames := self._inventory.get('areas', []):
            logger.info(f"Loading areas from '{filenames[0]}'.")
            with self.__open() as zfp:
                _, areas = self.__load(zfp, filenames[0])
                return areas
        else:
            logger.debug("No areas file found in archive.")
            return None

    def lazy_load_graphs(self) -> list[str]:
        """Return the list of graph filenames present in the archive.

        Returns
        -------
        list[str]
            Filenames that can be passed to :meth:`load_graph`.
        """
        return self._inventory.get('graphs', [])

    def load_graphs(self, areas_desc: pd.DataFrame) -> dict[str, SpatioTemporalGraph]:
        """Load all graphs from the archive.

        Parameters
        ----------
        areas_desc : pd.DataFrame
            Areas descriptor data frame required for graph deserialization.

        Returns
        -------
        dict[str, SpatioTemporalGraph]
            Mapping from graph name to its deserialized object.
        """
        filenames = self.lazy_load_graphs()
        logger.info(f"Loading {len(filenames)} graph(s) from '{self.filepath}'.")
        graphs = {}
        with self.__open() as zfp:
            for filename in filenames:
                name, graph = self.__load(zfp, filename, areas=areas_desc)
                graphs[name] = graph
        return graphs

    def load_graph(self, areas_desc: pd.DataFrame, filename: str) -> Optional[SpatioTemporalGraph]:
        """Load a single graph by its filename from the archive.

        Parameters
        ----------
        areas_desc : pd.DataFrame
            Areas descriptor data frame required for graph deserialization.
        filename : str
            Filename of the graph entry inside the ZIP archive.

        Returns
        -------
        SpatioTemporalGraph or None
            The deserialized graph, or ``None`` if *filename* is not in the archive.
        """
        if filename not in self.lazy_load_graphs():
            logger.debug(f"Graph '{filename}' not found in archive.")
            return None

        logger.info(f"Loading graph '{filename}' from '{self.filepath}'.")
        with self.__open() as zfp:
            _, graph = self.__load(zfp, filename, areas=areas_desc)
            return graph

    def lazy_load_matrices(self) -> list[str]:
        """Return the list of matrix filenames present in the archive.

        Returns
        -------
        list[str]
            Filenames that can be passed to :meth:`load_matrix`.
        """
        return self._inventory.get('matrices', [])

    def load_matrices(self) -> dict[str, np.ndarray]:
        """Load all matrices from the archive.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from matrix name to its deserialized array.
        """
        filenames = self.lazy_load_matrices()
        logger.info(f"Loading {len(filenames)} matri(ces) from '{self.filepath}'.")
        matrices = {}
        with self.__open() as zfp:
            for filename in filenames:
                name, matrix = self.__load(zfp, filename)
                matrices[name] = matrix
        return matrices

    def load_matrix(self, filename: str) -> Optional[np.ndarray]:
        """Load a single matrix by its filename from the archive.

        Parameters
        ----------
        filename : str
            Filename of the matrix entry inside the ZIP archive.

        Returns
        -------
        np.ndarray or None
            The deserialized array, or ``None`` if *filename* is not in the archive.
        """
        if filename not in self.lazy_load_matrices():
            logger.debug(f"Matrix '{filename}' not found in archive.")
            return None

        logger.info(f"Loading matrix '{filename}' from '{self.filepath}'.")
        with self.__open() as zfp:
            _, matrix = self.__load(zfp, filename)
            return matrix

    def lazy_load_metrics(self) -> list[str]:
        """Return the list of metrics filenames present in the archive.

        Returns
        -------
        list[str]
            Filenames that can be passed to :meth:`load_metric`.
        """
        return self._inventory.get('metrics', [])

    def load_metrics(self) -> dict[str, pd.DataFrame]:
        """Load all metric data frames from the archive.

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping from metric name to its deserialized data frame.
        """
        filenames = self.lazy_load_metrics()
        logger.info(f"Loading {len(filenames)} metric(s) from '{self.filepath}'.")
        metrics = {}
        with self.__open() as zfp:
            for filename in filenames:
                name, metric = self.__load(zfp, filename)
                metrics[name] = metric
        return metrics

    def load_metric(self, filename: str) -> Optional[pd.DataFrame]:
        """Load a single metric data frame by its filename from the archive.

        Parameters
        ----------
        filename : str
            Filename of the metric entry inside the ZIP archive.

        Returns
        -------
        pd.DataFrame or None
            The deserialized data frame, or ``None`` if *filename* is not in the archive.
        """
        if filename not in self.lazy_load_metrics():
            logger.debug(f"Metric '{filename}' not found in archive.")
            return None

        logger.info(f"Loading metric '{filename}' from '{self.filepath}'.")
        with self.__open() as zfp:
            _, metrics = self.__load(zfp, filename)
            return metrics

    def lazy_load_frequent_patterns(self) -> list[str]:
        """Return the list of frequent pattern filenames present in the archive.

        Returns
        -------
        list[str]
            Filenames that can be passed to :meth:`load_frequent_pattern`.
        """
        return self._inventory.get('frequent_patterns', [])



    def load_frequent_patterns(self) -> dict[tuple[str, str], FrequentPatterns]:
        """Load all frequent pattern dicts from the archive.

        Returns
        -------
        dict[tuple[str,str], FrequentPatterns]
            Mapping from ``(subject, mode)`` to the patterns object.
        """
        filenames = self.lazy_load_frequent_patterns()
        logger.info(f"Loading {len(filenames)} frequent pattern file(s) from '{self.filepath}'.")
        patterns = {}
        with self.__open() as zfp:
            for filename in filenames:
                name, pattern = self.__load(zfp, filename)
                patterns[tuple(name.split(','))] = pattern
        return patterns

    def load_frequent_pattern(self, filename: str) -> Optional[FrequentPatterns]:
        """Load a single frequent pattern dict by its filename from the archive.

        Parameters
        ----------
        filename : str
            Filename of the pattern entry inside the ZIP archive.

        Returns
        -------
        FrequentPatterns or None
            The pattern dict, or ``None`` if *filename* is not in the archive.
        """
        if filename not in self.lazy_load_frequent_patterns():
            logger.debug(f"Frequent pattern file '{filename}' not found in archive.")
            return None
        logger.info(f"Loading frequent pattern '{filename}' from '{self.filepath}'.")
        with self.__open() as zfp:
            _, pattern = self.__load(zfp, filename)
            return pattern


@dataclass(frozen=True)
class DataSaver:
    """Accumulates data items in memory and writes them to a ZIP archive.

    Items are staged via ``add_*`` methods and flushed to disk by calling
    :meth:`save`.  If the target archive already exists, only files whose
    names overlap with the new data are replaced; all other existing entries
    are preserved.
    """

    # TODO think about zip compression to gain space (is it fast enough?)

    _inventory: dict[str, list[tuple[str, Any]]] = field(default_factory=lambda: {})

    def __str__(self) -> str:
        print(self._inventory.keys())
        inventory = [f"{kind}: {len(item)} item(s)" for kind, item in self._inventory.items()]
        return f"DataSaver(inventory=[{', '.join(inventory)}])"

    def __add(self, kind: str, item: Any) -> None:
        """Append *item* to the staging inventory under *kind*.

        Parameters
        ----------
        kind : str
            Handler kind label.
        item : Any
            ``(name, data)`` tuple to stage.
        """
        self._inventory.setdefault(kind, []).append(item)

    def add_areas(self, areas: pd.DataFrame) -> None:
        """Stage an areas descriptor data frame for saving.

        Parameters
        ----------
        areas : pd.DataFrame
            Areas descriptor data frame.
        """
        logger.debug("Staging areas data frame.")
        self.__add('areas', ('areas', areas))

    def add_graphs(self, graphs: dict[str, SpatioTemporalGraph]) -> None:
        """Stage a collection of graphs for saving.

        Parameters
        ----------
        graphs : dict[str, SpatioTemporalGraph]
            Mapping from graph name to graph object.
        """
        logger.debug(f"Staging {len(graphs)} graph(s): {list(graphs.keys())}.")
        for name, graph in graphs.items():
            self.__add('graphs', (name, graph))

    def add_matrices(self, matrices: dict[str, np.ndarray]) -> None:
        """Stage a collection of NumPy matrices for saving.

        Parameters
        ----------
        matrices : dict[str, np.ndarray]
            Mapping from matrix name to array.
        """
        logger.debug(f"Staging {len(matrices)} matri(ces): {list(matrices.keys())}.")
        for name, matrix in matrices.items():
            self.__add('matrices', (name, matrix))

    def add_metrics(self, metrics: dict[str, pd.DataFrame]) -> None:
        """Stage a collection of metric data frames for saving.

        Parameters
        ----------
        metrics : dict[str, pd.DataFrame]
            Mapping from metric name to data frame.
        """
        logger.debug(f"Staging {len(metrics)} metric(s): {list(metrics.keys())}.")
        for name, metric in metrics.items():
            self.__add('metrics', (name, metric))

    def add_frequent_patterns(self, patterns: dict[tuple[str, str], FrequentPatterns]) -> None:
        """Stage frequent pattern dicts for saving.

        Parameters
        ----------
        patterns : dict mapping ``name`` to patterns dict.
        """
        logger.debug(f"Staging {len(patterns)} frequent patterns: {list(patterns.keys())}.")
        for name, pattern in patterns.items():
            self.__add('frequent_patterns', (','.join(name), pattern))

    @staticmethod
    def __save(zfp: ZipFile, filename: str, item: Any) -> None:
        """Serialize and write a single item to an open ZIP archive.

        Parameters
        ----------
        zfp : ZipFile
            Open ZIP archive in write or append mode.
        filename : str
            Name of the entry to create inside the archive.
        item : Any
            Data object to serialize.
        """
        logger.debug(f"Writing '{filename}' to archive.")
        fileinfo = ZipInfo(filename, date_time=datetime.datetime.now().timetuple()[:6])
        with zfp.open(fileinfo, 'w') as fp:
            try:
                DataRegistry.serialize(filename, item, fp)
            except NoDataHandlerFound as e:
                logger.error(f"Unable to save item \"{filename}\": {e}")

    def __gather_data(self) -> dict[str, Any]:
        """Flatten the staging inventory into a ``{filename: item}`` mapping.

        Returns
        -------
        dict[str, Any]
            Mapping from on-disk filename to the corresponding data object.
        """
        data = {}
        for kind, items in self._inventory.items():
            for name, item in items:
                filename = DataRegistry.name2filename(name, kind)
                data[filename] = item
        logger.debug(f"Gathered {len(data)} item(s) to save: {list(data.keys())}.")
        return data

    @staticmethod
    def __find_common_filenames(filepath: Path, data: dict[str, Any]) -> set[str]:
        """Return filenames present both in an existing archive and in *data*.

        Parameters
        ----------
        filepath : Path
            Path to an existing ZIP archive, or a non-existent path.
        data : dict[str, Any]
            Mapping of filenames about to be written.

        Returns
        -------
        set[str]
            Filenames that would be overwritten.  Empty if *filepath* does not exist.
        """
        if not filepath.exists():
            return set()

        with ZipFile(str(filepath)) as zfp:
            common = set(zfp.namelist()) & set(data.keys())
        if common:
            logger.debug(f"Found {len(common)} overlapping filename(s) in existing archive: {common}.")
        return common

    def __transfer_save(self, filepath: Path, data: dict[str, Any], common_filenames: set[str]) -> Generator[str, None, None]:
        """Replace overlapping entries in an existing archive, preserving others.

        The strategy is to write all unchanged entries to a temporary file,
        append the new data, then atomically replace the original archive.

        This method returns a generator yield, so it is required to iterate on so
        the saving to dataset can happen.

        Parameters
        ----------
        filepath : Path
            Path to the existing ZIP archive to update.
        data : dict[str, Any]
            Mapping from filename to data object for all items to write.
        common_filenames : set[str]
            Subset of filenames in *data* that already exist in the archive
            and must be replaced.

        Yields
        ------
        str
            The name of the item being saved
        """
        logger.info(f"Updating archive '{filepath}': replacing {len(common_filenames)} existing "
                    f"file(s) and adding new ones.")
        with tempfile.NamedTemporaryFile(suffix='.zip', delete_on_close=False) as tmp:
            # copy unchanged files
            with ZipFile(str(filepath), 'r') as zfp_in, \
                    ZipFile(tmp, 'w') as zfp_out:
                for fileinfo in zfp_in.infolist():
                    if fileinfo.filename not in common_filenames:
                        logger.debug(f"Preserving unchanged entry '{fileinfo.filename}'.")
                        with zfp_in.open(fileinfo, 'r') as src, zfp_out.open(fileinfo, 'w') as dst:
                            dst.write(src.read())
                            yield fileinfo.filename

            # add new files
            with ZipFile(tmp, 'a') as zfp:
                for filename, item in data.items():
                    self.__save(zfp, filename, item)
                    yield filename

            # replace old zip with new one
            logger.debug(f"Replacing '{filepath}' with updated archive.")
            Path(tmp.name).replace(filepath)

    def __simple_save(self, filepath: Path, data: dict[str, Any]) -> Generator[str, None, None]:
        """Append all items in *data* to a ZIP archive (creating it if necessary).

        This method returns a generator yield, so it is required to iterate on so
        the saving to dataset can happen.

        Parameters
        ----------
        filepath : Path
            Path to the destination ZIP archive.
        data : dict[str, Any]
            Mapping from filename to data object.

        Yields
        ------
        str
            The name of the item being saved
        """
        logger.info(f"Appending {len(data)} item(s) to archive '{filepath}'.")
        with ZipFile(str(filepath), 'a') as zfp:
            for filename, item in data.items():
                self.__save(zfp, filename, item)
                yield filename

    @staticmethod
    def __nb_files(filepath: Path) -> int:
        with ZipFile(str(filepath), 'r') as zfp:
            return len(zfp.namelist())

    def save(self, filepath: Path) -> tuple[int, Generator[str, None, None]]:
        """Flush all staged items to a ZIP archive at *filepath*.

        If the archive already exists and some staged filenames collide with
        existing entries, :meth:`__transfer_save` is used to replace only those
        entries while preserving the rest.  Otherwise, new entries are simply
        appended.

        This method returns a generator yield, so it is required to iterate on so
        the saving to dataset can happen.

        Parameters
        ----------
        filepath : Path
            Destination ZIP archive path.

        Returns
        ------
        tuple[int, Generator[str, None, None]]
            The total number of items to save and a generator to the saving of
            items that yield the name of the item being saved.
        """
        logger.info(f"Saving data to {filepath}.")
        data = self.__gather_data()
        if common_filenames := self.__find_common_filenames(filepath, data):
            nb_files_in_dataset = self.__nb_files(filepath)
            total_nb_files = nb_files_in_dataset - len(common_filenames) + len(data)
            return total_nb_files, self.__transfer_save(filepath, data, common_filenames)
        else:
            return len(data), self.__simple_save(filepath, data)


def load_spatio_temporal_graph(filepath: Path | str) -> SpatioTemporalGraph:
    """Load a spatio-temporal graph from its zip file.

    If multiple graphs are in the archive, the first found will be loaded.

    Parameters
    ----------
    filepath: Path | str
        The path to the zip file.

    Returns
    -------
    SpatioTemporalGraph
        The spatio-temporal graph contained in the zip file.

    Example
    -------
    >>> G = nx.DiGraph()
    >>> G.add_nodes_from({
    ...     1: {'t': 0, 'areas': {1}, 'region': 'R1', 'internal_strength': 1},
    ...     2: {'t': 0, 'areas': {2}, 'region': 'R1', 'internal_strength': 1},
    ...     3: {'t': 0, 'areas': {3}, 'region': 'R2', 'internal_strength': 1},
    ...     4: {'t': 1, 'areas': {1, 2}, 'region': 'R1', 'internal_strength': 0.52873788},
    ...     5: {'t': 1, 'areas': {3}, 'region': 'R2', 'internal_strength': 1}})
    >>> G.add_edges_from([
    ...     (1, 3, {'t': 0, 'type': 'spatial', 'correlation': -0.41853318}),
    ...     (1, 4, {'type': 'temporal', 'transition': RC5.PP}),
    ...     (2, 3, {'t': 0, 'type': 'spatial', 'correlation': 0.75087697}),
    ...     (2, 4, {'type': 'temporal', 'transition': RC5.PP}),
    ...     (3, 1, {'t': 0, 'type': 'spatial', 'correlation': -0.41853318}),
    ...     (3, 2, {'t': 0, 'type': 'spatial', 'correlation': 0.75087697}),
    ...     (3, 5, {'type': 'temporal', 'transition': RC5.EQ}),
    ...     (4, 5, {'t': 1, 'type': 'spatial', 'correlation': 0.75087697}),
    ...     (5, 4, {'t': 1, 'type': 'spatial', 'correlation': 0.75087697})])
    >>> areas_desc = pd.DataFrame({
    ...     'Id_Area': [1, 2, 3],
    ...     'Name_Area': ['Area 1', 'Area 2', 'Area 3'],
    ...     'Name_Region': ['R1', 'R2', 'R3']})
    >>> areas_desc.set_index('Id_Area', inplace=True)
    >>> graph_path = Path('/tmp/tmp.zip')
    >>> save_spatio_temporal_graph(SpatioTemporalGraph(G, areas_desc), graph_path)
    >>> graph_struct = load_spatio_temporal_graph(graph_path)

    Raises
    ------
    RuntimeError
        If no graph is found in the zip file.
    """
    logger.debug(f"Loading STG from '{filepath}'.")
    loader = DataLoader(filepath)
    filenames = loader.lazy_load_graphs()

    if len(filenames) > 0:
        filename = filenames[0]
    else:
        raise RuntimeError("No graph found in data file.")

    areas = loader.load_areas()
    return loader.load_graph(areas, filename)


def save_spatio_temporal_graph(graph: SpatioTemporalGraph, filepath: Path | str) -> None:
    """Save a spatio-temporal graph to a zip file.

    Parameters
    ----------
    graph: SpatioTemporalGraph
        The spatio-temporal graph to save.
    filepath: Path | str
        THe path to the zip file.

    Example
    -------
    >>> G = nx.DiGraph()
    >>> G.add_nodes_from({
    ...     1: {'t': 0, 'areas': {1}, 'region': 'R1', 'internal_strength': 1},
    ...     2: {'t': 0, 'areas': {2}, 'region': 'R1', 'internal_strength': 1},
    ...     3: {'t': 0, 'areas': {3}, 'region': 'R2', 'internal_strength': 1},
    ...     4: {'t': 1, 'areas': {1, 2}, 'region': 'R1', 'internal_strength': 0.52873788},
    ...     5: {'t': 1, 'areas': {3}, 'region': 'R2', 'internal_strength': 1}})
    >>> G.add_edges_from([
    ...     (1, 3, {'t': 0, 'type': 'spatial', 'correlation': -0.41853318}),
    ...     (1, 4, {'type': 'temporal', 'transition': RC5.PP}),
    ...     (2, 3, {'t': 0, 'type': 'spatial', 'correlation': 0.75087697}),
    ...     (2, 4, {'type': 'temporal', 'transition': RC5.PP}),
    ...     (3, 1, {'t': 0, 'type': 'spatial', 'correlation': -0.41853318}),
    ...     (3, 2, {'t': 0, 'type': 'spatial', 'correlation': 0.75087697}),
    ...     (3, 5, {'type': 'temporal', 'transition': RC5.EQ}),
    ...     (4, 5, {'t': 1, 'type': 'spatial', 'correlation': 0.75087697}),
    ...     (5, 4, {'t': 1, 'type': 'spatial', 'correlation': 0.75087697})])
    >>> areas_desc = pd.DataFrame({
    ...     'Name_Area': ['Area 1', 'Area 2', 'Area 3'],
    ...     'Name_Region': ['R1', 'R2', 'R3']}, index=[1, 2, 3])
    >>> graph_path = Path('/tmp/tmp.zip')
    >>> graph_struct = SpatioTemporalGraph(G, areas_desc)
    >>> save_spatio_temporal_graph(graph_struct, graph_path)
    """
    logger.debug(f"Saving STG to '{filepath}'.")
    saver = DataSaver()
    saver.add_areas(graph.areas)
    saver.add_graphs({'graph.json': graph})
    _, gen = saver.save(filepath)
    list(gen)


class FrequentPatternsIO:
    """I/O utilities for loading frequent subgraph patterns from SPMiner output files.

    Provides class methods for loading frequent patterns from JSON files generated by the SPMiner
    service. Patterns are parsed into FrequentPattern objects (networkx DiGraph subclasses) and
    aggregated into ``FrequentPatterns`` collections.
    """

    @classmethod
    def from_spminer_file(cls, file: Path) -> 'FrequentPatterns':
        """Load frequent patterns from a single SPMiner JSON output file.

        Parses a JSON file containing frequent subgraph patterns and returns a FrequentPatterns
        collection with all patterns decoded from their dictionary representation.

        Parameters
        ----------
        file : Path
            Path to a JSON file containing frequent patterns from SPMiner output.
            Expected format: dict mapping pattern names to pattern dictionaries with
            'nodes' and 'edges' keys.

        Returns
        -------
        FrequentPatterns
            A FrequentPatterns dataclass containing the parsed patterns.

        Examples
        --------
        >>> patterns = FrequentPatternsIO.from_spminer_file(Path('motifs_enriched_s.json'))
        >>> len(patterns)  # doctest: +SKIP
        5
        """
        with open(file, 'r') as fp:
            json_data = json.load(fp)
            patterns = {name: FrequentPattern.from_dict(pattern) for name, pattern in json_data.items()}
            return FrequentPatterns(patterns)

    @classmethod
    def from_spminer_files(cls, output_dir: Path, filenames: Iterable[Path]) -> dict[tuple[str, str], 'FrequentPatterns']:
        """Load frequent patterns from multiple SPMiner JSON output files.

        Loads patterns from multiple files and returns a dictionary where keys are derived from
        relative file paths (directory structure and filename without extension), and values are
        FrequentPatterns collections.

        Parameters
        ----------
        output_dir : Path
            Base output directory relative to which file paths are computed for the result keys.
        filenames : Iterable of Path
            Iterable of paths to JSON files containing frequent patterns from SPMiner output.

        Returns
        -------
        dict[str, FrequentPatterns]
            Mapping from relative file paths (without extension) to FrequentPatterns objects.
            For example, a file at ``output_dir/subject_A/motifs_enriched_s.json`` would be
            keyed as ``'subject_A/motifs_enriched_s'``.

        Examples
        --------
        >>> patterns_dict = FrequentPatternsIO.from_spminer_files(  # doctest: +SKIP
        ...     Path('output'), [Path('output/subject_A/motifs_enriched_s.json')]
        ... )
        >>> 'subject_A/motifs_enriched_s' in patterns_dict  # doctest: +SKIP
        True
        """
        all_patterns = {}
        for filename in filenames:
            try:
                name = DataRegistry.filename2name(str(filename.relative_to(output_dir)))
                subject, mode = name.split(',')
                all_patterns[subject, mode] = cls.from_spminer_file(filename)
            except Exception as ex:
                logger.debug(f"Skipping {filename}: {ex}")
        return all_patterns


@dataclass(frozen=True)
class GraphsDataset:
    """Dataset for managing spatio-temporal graphs and associated (meta)data.

    The dataset is lazily loaded from a specified file path, which contains graph and matrix files.
    The dataset includes a description of areas (nodes) in the graphs, factors for grouping subjects,
    and a table of subjects with their associated graph and matrix files.

    Some methods allow for retrieving graphs and matrices by subject IDs,
    checking for the presence of matrices, and serializing/deserializing the dataset.

    Attributes
    ----------
    loader : DataLoader
        Loader object for reading graph and matrix files.
    areas_desc : pandas.DataFrame
        A dataframe describing the areas (nodes) in the graphs.
    factors : list of set of str
        List of sets, each containing factor names for grouping subjects.
    subjects : pandas.DataFrame
        A dataframe containing subject information, indexed by factors and subject ID.

    Methods
    -------
    serialize() -> Dict[str, Any]
        Serializes the dataset into a dictionary format.
    get_graph(ids: Tuple[str, ...]) -> SpatioTemporalGraph
        Retrieves the graph associated with the given subject IDs.
    has_matrices() -> bool
        Checks if the dataset contains matrices for subjects.
    get_matrix(ids: Tuple[str, ...]) -> np.ndarray
        Retrieves the matrix associated with the given subject IDs.
    deserialize(data: Dict[str, Any]) -> 'GraphsDataset'
        Deserializes a dataset from a dictionary format.
    from_filepath(filepath: Path) -> 'GraphsDataset'
        Creates a GraphsDataset instance from a file path, loading the dataset lazily.
    """

    loader: DataLoader
    areas_desc: pd.DataFrame
    factors: list[set[str]]
    subjects: pd.DataFrame

    def serialize(self) -> dict[str, Any]:
        """Serializes the dataset into a dictionary format for storage or transmission.

        Returns
        -------
        A dictionary mapping dataset attributes to their values, including:
        - 'filepath': The file path of the dataset.
        - 'areas_desc': A list of dictionaries representing the areas description.
        - 'factors': A list of sets, each containing factor names.
        - 'subjects': A list of dictionaries representing the subjects table.
        """
        return {
            'filepath': str(self.loader.filepath),
            'areas_desc': self.areas_desc.reset_index().to_dict('records'),
            'factors': [list(f) for f in self.factors],
            'subjects': self.subjects.reset_index().to_dict('records')
        }

    def __contains__(self, ids: tuple[str, ...]) -> bool:
        return ids in self.subjects.index

    def get_graph(self, ids: tuple[str, ...]) -> SpatioTemporalGraph:
        """Retrieves the graph associated with the given subject IDs.

        Parameters
        ----------
        ids : tuple of str
            A tuple of strings representing the subject IDs, which should match
            the index of the subjects Data.

        Returns
        -------
        SpatioTemporalGraph
            The spatio-temporal graph corresponding to the specified subject IDs.

        Raises
        ------
        KeyError
            If the provided IDs do not match any subject in the dataset.
        """
        filename = self.subjects.loc[ids]['Graph']
        return self.loader.load_graph(self.areas_desc, filename)

    def has_matrices(self) -> bool:
        """Checks if the dataset contains matrices for subjects.

        Returns
        -------
        bool
            True if the dataset has a 'Matrix' column in the subjects DataFrame, False otherwise.
        """
        return 'Matrix' in self.subjects.columns

    def get_matrix(self, ids: tuple[str, ...]) -> np.ndarray:
        """Retrieves the matrix associated with the given subject IDs.

        Parameters
        ----------
        ids : tuple of str
            A tuple of strings representing the subject IDs, which should match
            the index of the subjects DataFrame.

        Returns
        -------
        numpy.ndarray
            The matrix corresponding to the specified subject IDs.

        Raises
        ------
        KeyError
            If the provided IDs do not match any subject in the dataset.
        """
        filename = self.subjects.loc[ids]['Matrix']
        return self.loader.load_matrix(filename)

    def get_available_metrics(self) -> list[str]:
        return list(self.loader.load_metrics().keys())

    def has_metrics(self, name: Optional[str] = None) -> bool:
        available_metrics = self.get_available_metrics()
        return name and name in available_metrics or len(available_metrics) > 0

    def get_metrics(self, name: str) -> Optional[pd.DataFrame]:
        return self.loader.load_metrics()[name]

    def has_frequent_patterns(self) -> bool:
        """Check if the dataset contains frequent pattern files.

        Returns
        -------
        bool
            ``True`` if at least one frequent pattern file is present.
        """
        return len(self.loader.lazy_load_frequent_patterns()) > 0

    def get_available_frequent_pattern_modes(self) -> list[str]:
        """Return sorted list of available frequent pattern mining modes.

        Parses filenames from :meth:`DataLoader.lazy_load_frequent_patterns`
        using the :attr:`FrequentPatternsHandler.pattern` regex and collects
        unique mode groups.

        Returns
        -------
        list of str
            Sorted unique mode identifiers (e.g. ``['s', 'st', 't']``).
        """
        modes: set[str] = set()
        for filename in self.loader.lazy_load_frequent_patterns():
            try:
                name = DataRegistry.filename2name(filename)
                _, mode = name.split(',')
                modes.add(mode)
            except Exception as ex:
                logger.debug(f"Skipping {filename}: {ex}")

        return sorted(modes)

    def get_frequent_patterns(self, ids: tuple[str, ...]) -> list[FrequentPatterns]:
        """Get frequent patterns for a subject.

        Parameters
        ----------
        ids : tuple[str, ...]
            Subject index values (same as used by :meth:`get_graph`).

        Returns
        -------
        list[FrequentPatterns]
            The objects to manipulate frequent patterns for each mode or an empty list.
        """
        graph_filename = self.subjects.loc[ids]['Graph']
        subject_dir = str(Path(graph_filename).with_suffix(''))

        if filenames := [filename for filename in self.loader.lazy_load_frequent_patterns()
                         if filename.startswith(subject_dir)]:
            return [self.loader.load_frequent_pattern(filename) for filename in filenames]
        else:
            return []

    @staticmethod
    def deserialize(data: dict[str, Any]) -> 'GraphsDataset':
        """Deserializes a dataset from a dictionary format.

        Parameters
        ----------
        data : dict mapping property names to their values
            A dictionary containing the serialized dataset properties, including:
            - 'filepath': The file path of the dataset.
            - 'areas_desc': A list of dictionaries representing the areas description.
            - 'factors': A list of sets, each containing factor names.
            - 'subjects': A list of dictionaries representing the subjects table.

        Returns
        -------
        GraphsDataset
            An instance of GraphsDataset created from the provided data.
        """
        filepath = Path(data['filepath'])
        n = len(data['factors'])
        subjects = pd.DataFrame(data['subjects'])
        return GraphsDataset(loader=DataLoader(filepath),
                             areas_desc=pd.DataFrame(data['areas_desc']).set_index('Id_Area'),
                             factors=[set(f) for f in data['factors']],
                             subjects=subjects.set_index(list(subjects.columns[:n+1])))

    @staticmethod
    def from_filepath(filepath: Path) -> 'GraphsDataset':
        """Creates a GraphsDataset instance from a file path, loading the dataset lazily.

        Parameters
        ----------
        filepath : pathlib.Path
            The path to the dataset file, which should contain graph and matrix files.

        Returns
        -------
        GraphsDataset
            An instance of GraphsDataset created from the specified file path.
        """
        # load the dataset lazily
        loader = DataLoader(filepath)
        areas_desc = loader.load_areas()
        graphs_filenames = loader.lazy_load_graphs()
        matrices_filenames = loader.lazy_load_matrices()

        if areas_desc is None or graphs_filenames is None or matrices_filenames is None:
            raise IOError("No dataset red.")

        # extract factors from filename (without extension
        filenames_without_ext = [name.split('.json')[0] for name in graphs_filenames]
        factors: list[set[str]]
        factors, ids = split_factors_from_name(filenames_without_ext)

        # create a subject's table with factors as index and filenames as data
        data = list(zip(*zip(*ids), graphs_filenames))
        n = len(factors)
        columns = [f'Factor{i+1}' for i in range(n)] + ['Subject', 'Graph']
        subjects = pd.DataFrame(data, columns=columns).set_index(columns[:n+1])

        if len(matrices_filenames) == len(graphs_filenames):
            subjects['Matrix'] = matrices_filenames

        return GraphsDataset(loader=loader,
                             areas_desc=areas_desc,
                             factors=factors,
                             subjects=subjects)

    def __str__(self) -> str:
        return f"GraphsDataset(filepath=\"{self.loader.filepath}\", "\
               f"#areas={len(self.areas_desc)}, #subjects={len(self.subjects)}, "\
               f"factors={self.factors})"

    def __repr__(self) -> str:
        return str(self)
