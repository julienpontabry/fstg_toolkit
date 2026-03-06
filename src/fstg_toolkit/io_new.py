import datetime
import json
import logging
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, IO, Protocol, Optional, Generator
from zipfile import ZipFile, ZipInfo

import networkx as nx
import numpy as np
import pandas as pd

from fstg_toolkit import SpatioTemporalGraph
from fstg_toolkit.graph import RC5
from fstg_toolkit.io import _SpatioTemporalGraphEncoder, _spatio_temporal_object_hook

logger = logging.getLogger()


class DataHandler(Protocol):
    def matches(self, filename: str) -> bool:
        """Check if the handler can handle a file from its filename."""

    def serialize(self, item: Any, fp: IO, **context: Any) -> None:
        """Serialize the item to a file-like object."""

    def deserialize(self, fp: IO, **context: Any) -> Any:
        """Deserialize the item from a file-like object."""

    def filename2name(self, filename: str) -> str:
        """Convert a filename to its corresponding name."""

    def name2filename(self, name: str) -> str:
        """Convert a name to its corresponding filename."""


class NoDataHandlerFound(TypeError):
    def __init__(self, name: str) -> None:
        super().__init__(f"No handler found for \"{name}\".")


class DataRegistry:
    _handlers: dict[str, DataHandler] = {}

    @classmethod
    def register(cls, kind: str):
        def decorator(handler_cls):
            cls._handlers[kind] = handler_cls
            return handler_cls
        return decorator

    @classmethod
    def resolve(cls, filename: str) -> Optional[tuple[str, DataHandler]]:
        """Find the handler matching the given filename."""
        for kind, handler in cls._handlers.items():
            if handler.matches(filename):
                return kind, handler
        return None

    @classmethod
    def classify(cls, filenames: list[str]) -> dict[str, list[str]]:
        results = {}
        for filename in filenames:
            if resolved := cls.resolve(filename):
                kind, _ = resolved
                results.setdefault(kind, []).append(filename)
        return results

    @classmethod
    def name2filename(cls, name: str, kind: str) -> str:
        if handler := cls._handlers.get(kind, None):
            return handler.name2filename(name)
        else:
            raise NoDataHandlerFound(name)

    @classmethod
    def serialize(cls, filename: str, item: Any, fp: IO, **context) -> None:
        if resolved := cls.resolve(filename):
            _, handler = resolved
            handler.serialize(item, fp, **context)
        else:
            raise NoDataHandlerFound(filename)


    @classmethod
    def deserialize(cls, filename: str, fp: IO, **context: Any) -> tuple[str, Any]:
        if resolved := cls.resolve(filename):
            _, handler = resolved
            item = handler.deserialize(fp, **context)
            name = handler.filename2name(filename)
            return name, item
        else:
            raise NoDataHandlerFound(filename)


@DataRegistry.register('areas')
class AreasDescHandler(Protocol):
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
                df[col] = df[col].apply(
                    lambda x: json.dumps({k.name: v for k, v in x.items()}) if isinstance(x, dict) else x)

        # serialize multi-index if any
        # NOTE not compatible with floating points index elements
        if isinstance(df.index, pd.MultiIndex):
            df.index = pd.Index(['.'.join([str(e) for e in idx]) for idx in df.index],
                                name='.'.join(df.index.names))

        # serialize multi-columns if any
        if isinstance(df.columns, pd.MultiIndex):
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
            if any(isinstance(x, str) and x.strip().startswith('{') and x.strip().endswith('}')
                   for x in df[col].dropna()):
                df[col] = df[col].apply(lambda x: MetricsHandler.__to_rc5_if_possible(json.loads(x))
                if isinstance(x, str) and x.strip().startswith('{') and x.strip().endswith('}') else x)

        # deserialize multi-index if any
        if '.' in df.index.name and all('.' in idx for idx in df.index):
            tuples = [tuple(int(i) if i.isdigit() else i
                             for i in idx.split('.'))
                      for idx in df.index]
            df.index = pd.MultiIndex.from_tuples(tuples, names=df.index.name.split('.'))

        # deserialize multi-columns if any
        if all('.' in c for c in df.columns):
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


@dataclass(frozen=True)
class DataLoader:
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

        with self.__open() as zfp:
            self._inventory.update(DataRegistry.classify(zfp.namelist()))

    def __str__(self) -> str:
        inventory = [f"{kind}: {len(filenames)} filename(s) {"(ex: " + filenames[0] + ")" if filenames else ""}"
                     for kind, filenames in self._inventory.items()]
        return f"DataLoader(filepath={self.filepath}, inventory=[{', '.join(inventory)}])"

    @contextmanager
    def __open(self) -> Generator[ZipFile, None, None]:
        with ZipFile(self.filepath) as zfp:
            yield zfp

    @staticmethod
    def __load(zfp: ZipFile, filename: str, **context: Any) -> Optional[Any]:
        with zfp.open(filename) as fp:
            try:
                return DataRegistry.deserialize(filename, fp, **context)
            except NoDataHandlerFound as e:
                logger.error(f"Unable to load item \"{filename}\": {e}.")

    def load_areas(self) -> Optional[pd.DataFrame]:
        if filenames := self._inventory.get('areas', []):
            with self.__open() as zfp:
                _, areas = self.__load(zfp, filenames[0])
                return areas
        else:
            return None

    def lazy_load_graphs(self) -> list[str]:
        return self._inventory.get('graphs', [])

    def load_graphs(self, areas_desc: pd.DataFrame) -> dict[str, SpatioTemporalGraph]:
        graphs = {}
        with self.__open() as zfp:
            for filename in self.lazy_load_graphs():
                name, graph = self.__load(zfp, filename, areas=areas_desc)
                graphs[name] = graph
        return graphs

    def load_graph(self, areas_desc: pd.DataFrame, filename: str) -> Optional[SpatioTemporalGraph]:
        if filename not in self.lazy_load_graphs():
            return None

        with self.__open() as zfp:
            _, graph = self.__load(zfp, filename, areas=areas_desc)
            return graph

    def lazy_load_matrices(self) -> list[str]:
        return self._inventory.get('matrices', [])

    def load_matrices(self) -> dict[str, np.ndarray]:
        matrices = {}
        with self.__open() as zfp:
            for filename in self.lazy_load_matrices():
                name, matrix = self.__load(zfp, filename)
                matrices[name] = matrix
        return matrices

    def load_matrix(self, filename: str) -> Optional[np.ndarray]:
        if filename not in self.lazy_load_matrices():
            return None

        with self.__open() as zfp:
            _, matrix = self.__load(zfp, filename)
            return matrix

    def lazy_load_metrics(self) -> list[str]:
        return self._inventory.get('metrics', [])

    def load_metrics(self) -> dict[str, pd.DataFrame]:
        metrics = {}
        with self.__open() as zfp:
            for filename in self.lazy_load_metrics():
                name, metric = self.__load(zfp, filename)
                metrics[name] = metric
        return metrics

    def load_metric(self, filename: str) -> Optional[pd.DataFrame]:
        if filename not in self.lazy_load_metrics():
            return None

        with self.__open() as zfp:
            _, metrics = self.__load(zfp, filename)
            return metrics


@dataclass(frozen=True)
class DataSaver:
    _inventory: dict[str, list[tuple[str, Any]]] = field(default_factory=lambda: {})

    def __str__(self) -> str:
        print(self._inventory.keys())
        inventory = [f"{kind}: {len(item)} item(s)" for kind, item in self._inventory.items()]
        return f"DataSaver(inventory=[{', '.join(inventory)}])"

    def __add(self, kind: str, item: Any) -> None:
        self._inventory.setdefault(kind, []).append(item)

    def add_areas(self, areas: pd.DataFrame) -> None:
        self.__add('areas', ('areas', areas))

    def add_graphs(self, graphs: dict[str, SpatioTemporalGraph]) -> None:
        for name, graph in graphs.items():
            self.__add('graphs', (name, graph))

    def add_matrices(self, matrices: dict[str, np.ndarray]) -> None:
        for name, matrix in matrices.items():
            self.__add('matrices', (name, matrix))

    def add_metrics(self, metrics: dict[str, pd.DataFrame]) -> None:
        for name, metric in metrics.items():
            self.__add('metrics', (name, metric))

    @staticmethod
    def __save(zfp: ZipFile, filename: str, item: Any) -> None:
        fileinfo = ZipInfo(filename, date_time=datetime.datetime.now().timetuple()[:6])
        with zfp.open(fileinfo, 'w') as fp:
            try:
                DataRegistry.serialize(filename, item, fp)
            except NoDataHandlerFound as e:
                logger.error(f"Unable to save item \"{filename}\": {e}.")

    def __gather_data(self) -> dict[str, Any]:
        data = {}
        for kind, items in self._inventory.items():
            for name, item in items:
                filename = DataRegistry.name2filename(name, kind)
                data[filename] = item
        return data

    @staticmethod
    def __find_common_filenames(filepath: Path, data: dict[str, Any]) -> set[str]:
        if not filepath.exists():
            return set()

        with ZipFile(str(filepath)) as zfp:
            return set(zfp.namelist()) & set(data.keys())

    def __transfer_save(self, filepath: Path, data: dict[str, Any], common_filenames: set[str]) -> None:
        with tempfile.NamedTemporaryFile(suffix='.zip', delete_on_close=False) as tmp:
            # copy unchanged files
            with ZipFile(str(filepath), 'r') as zfp_in, \
                    ZipFile(tmp, 'w') as zfp_out:
                for fileinfo in zfp_in.infolist():
                    if fileinfo.filename not in common_filenames:
                        with zfp_in.open(fileinfo, 'r') as src, zfp_out.open(fileinfo, 'w') as dst:
                            dst.write(src.read())

            # add new files
            with ZipFile(tmp, 'a') as zfp:
                for filename, item in data.items():
                    self.__save(zfp, filename, item)

            # replace old zip with new one
            Path(tmp.name).replace(filepath)

    def __simple_save(self, filepath: Path, data: dict[str, Any]) -> None:
        with ZipFile(str(filepath), 'a') as zfp:
            for filename, item in data.items():
                self.__save(zfp, filename, item)

    def save(self, filepath: Path) -> None:
        logger.info(f"Saving data to {filepath}.")
        data = self.__gather_data()
        if common_filenames := self.__find_common_filenames(filepath, data):
            self.__transfer_save(filepath, data, common_filenames)
        else:
            self.__simple_save(filepath, data)
