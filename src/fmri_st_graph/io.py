"""Defines helpers for inputs/outputs."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Dict, LiteralString, IO
from zipfile import ZipFile
import json

import numpy as np
import pandas as pd
import networkx as nx

from .graph import RC5, SpatioTemporalGraph


class _SpatioTemporalGraphEncoder(json.JSONEncoder):
    """JSON encoder for spatio-temporal graph.

    The sets are converted to lists and the RC5 objects are converted to their
    names as strings. The rest is left untouched.
    """

    def default(self, obj):
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
    saver = DataSaver()
    saver.add(graph.areas)
    saver.add({'graph.json': graph})
    saver.save(filepath)


def save_metrics(path_or_buf: str | Path | IO[bytes], metrics: pd.DataFrame) -> None:
    df: pd.DataFrame = metrics.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['.'.join(tuple(c)) for c in df.columns]

    if isinstance(df.index, pd.MultiIndex):
        df.index = pd.Index(['.'.join(tuple(idx)) for idx in df.index], name='.'.join(df.index.names))

    df.to_csv(path_or_buf)


def load_metrics(path_or_buf: str | Path | IO[bytes]) -> pd.DataFrame:
    df = pd.read_csv(path_or_buf, index_col=0)

    if '.' in df.index.name and all('.' in idx for idx in df.index):
        df.index = pd.MultiIndex.from_tuples([idx.split('.') for idx in df.index],
                                             names=df.index.name.split('.'))

    if all('.' in c for c in df.columns):
        df.columns = pd.MultiIndex.from_tuples([tuple(c.split('.')) for c in df.columns])

    return df


type SpatioTemporalGraphsDict = Dict[LiteralString, SpatioTemporalGraph]
type MatricesDict = Dict[LiteralString, np.ndarray]
type AnySpatioTemporalGraphsStruct = SpatioTemporalGraphsDict | List[LiteralString]
type GraphsLoader = Callable[[pd.DataFrame], AnySpatioTemporalGraphsStruct]
type AnyMatricesStruct = MatricesDict | List[LiteralString]
type MatricesLoader = Callable[[], AnyMatricesStruct]
type CorrelationMatricesDict = Dict[LiteralString, np.ndarray]


@dataclass
class DataLoader:
    """Loads spatio-temporal data (graph, areas description, raw matrices) from a zip archive.

    This class provides methods to load all or part of the data stored in a zip file, including:
    - Area descriptions (as a pandas DataFrame)
    - Spatio-temporal graphs (as SpatioTemporalGraph objects)
    - Correlation matrices (as numpy arrays)

    It supports both eager and lazy loading of graphs and matrices, and allows loading individual files by name.

    Attributes
    ----------
    filepath: pathlib.Path
        pathlib.Path to the zip archive containing the data.

    Methods
    -------
    load_areas() -> Optional[pd.DataFrame]
        Load the areas description DataFrame from the archive.
    load_graphs(areas: pd.DataFrame) -> SpatioTemporalGraphsDict
        Load all spatio-temporal graphs using the provided areas DataFrame.
    load_matrices() -> MatricesDict
        Load all matrices from the archive.
    lazy_load_graphs() -> List[LiteralString]
        List available graph filenames in the archive.
    lazy_load_matrices() -> List[LiteralString]
        List available matrix filenames in the archive.
    load_graph(areas: pd.DataFrame, filename: LiteralString) -> Optional[SpatioTemporalGraph]
        Load a single spatio-temporal graph by filename.
    load_matrix(filename: LiteralString) -> Optional[np.ndarray]
        Load a single matrix by filename.
    load() -> Optional[Tuple[pd.DataFrame, SpatioTemporalGraphsDict, MatricesDict]]
        Load areas, graphs, and matrices from the archive.
    lazy_load() -> Optional[Tuple[pd.DataFrame, List[LiteralString], List[LiteralString]]]
        Load areas and list available graph and matrix filenames.
    """
    filepath: Path

    @property
    def __within_archive(self):
        return ZipFile(str(self.filepath), 'r')

    def load_areas(self) -> Optional[pd.DataFrame]:
        """Load areas description DataFrame from the archive.

        Returns
        -------
        pandas.DataFrame or None
            A DataFrame containing areas description with 'Id_Area' as index, or None if the file is not found.
        """
        with self.__within_archive as zfp:
            with zfp.open('areas.csv', 'r') as fp:
                return pd.read_csv(fp, index_col='Id_Area')

    def load_graphs(self, areas: pd.DataFrame) -> SpatioTemporalGraphsDict:
        """Load all spatio-temporal graphs with the provided areas DataFrame.

        Parameters
        ----------
        areas: pandas.DataFrame
            A DataFrame containing areas description with 'Id_Area' as index.

        Returns
        -------
        dict[str, SpatioTemporalGraph]
            A dictionary of spatio-temporal graphs, where keys are graph names (filenames without '.json' extension)
            and values are SpatioTemporalGraph objects.
        """
        graphs = {}

        with self.__within_archive as zfp:
            for name in zfp.namelist():
                if name.endswith('.json'):
                    with zfp.open(name, 'r') as fp:
                        graph_dict = json.load(fp, object_hook=_spatio_temporal_object_hook)
                        graph = nx.json_graph.node_link_graph(graph_dict, edges='edges')
                        graphs[name.split('.json')[0]] = SpatioTemporalGraph(graph, areas)

        return graphs

    def load_matrices(self) -> MatricesDict:
        """Load all matrices from the archive.

        Returns
        -------
        dict[str, numpy.ndarray]
            A dictionary of matrices, where keys are matrix names (filenames without '.npy' extension)
            and values are numpy arrays representing the matrices.
        """
        matrices = {}

        with self.__within_archive as zfp:
            for name in zfp.namelist():
                if name.endswith('.npy'):
                    with zfp.open(name, 'r') as fp:
                        matrices[name.split('.npy')[0]] = np.load(fp)

        return matrices

    def __get_filenames(self, ext: LiteralString) -> List[LiteralString]:
        with self.__within_archive as zfp:
            return list(filter(lambda n: n.endswith(ext), zfp.namelist()))

    def lazy_load_graphs(self) -> List[LiteralString]:
        """List available graph filenames in the archive.

        Returns
        -------
        list[str]
            A list of filenames (with '.json' extension) of all spatio-temporal graphs.
        """
        return self.__get_filenames('.json')

    def lazy_load_matrices(self) -> List[LiteralString]:
        """List available matrix filenames in the archive.

        Returns
        -------
        list[str]
            A list of filenames (with '.npy' extension) of all matrices.
        """
        return self.__get_filenames('.npy')

    def load_graph(self, areas: pd.DataFrame, filename: LiteralString) -> Optional[SpatioTemporalGraph]:
        """Load a single spatio-temporal graph by filename.

        Parameters
        ----------
        areas: pandas.DataFrame
            A DataFrame containing areas description with 'Id_Area' as index.
        filename: str
            The name of the file (with '.json' extension) to load the graph from.

        Returns
        -------
        SpatioTemporalGraph or None
            A SpatioTemporalGraph object if the file is found and loaded successfully,
            or None if the file does not exist in the archive.
        """
        with self.__within_archive as zfp:
            with zfp.open(filename, 'r') as fp:
                graph_dict = json.load(fp, object_hook=_spatio_temporal_object_hook)
                graph = nx.json_graph.node_link_graph(graph_dict, edges='edges')
                return SpatioTemporalGraph(graph, areas)

    def load_matrix(self, filename: LiteralString) -> Optional[np.ndarray]:
        """Load a single matrix by filename.

        Parameters
        ----------
        filename: str
            The name of the file (with '.npy' extension) to load the matrix from

        Returns
        -------
        numpy.ndarray or None
            A numpy array representing the matrix if the file is found and loaded successfully,
            or None if the file does not exist in the archive.
        """
        with self.__within_archive as zfp:
            with zfp.open(filename, 'r') as fp:
                return np.load(fp)

    def __load_all_scheme(self, graphs_loader: GraphsLoader, matrices_loader: MatricesLoader) \
            -> Optional[Tuple[pd.DataFrame, AnySpatioTemporalGraphsStruct, AnyMatricesStruct]]:
        areas = self.load_areas()

        if areas is None:
            return None

        graphs = graphs_loader(areas)
        matrices = matrices_loader()

        return areas, graphs, matrices

    def load(self) -> Optional[Tuple[pd.DataFrame, SpatioTemporalGraphsDict, MatricesDict]]:
        """Load areas, graphs, and matrices from the archive.

        Returns
        -------
        tuple[pandas.DataFrame, dict[str, SpatioTemporalGraphs], dict[str, numpy.ndarray]] or None
            A tuple containing:
            - A DataFrame with areas description (index: 'Id_Area').
            - A dictionary of spatio-temporal graphs (keys: graph names, values: SpatioTemporalGraph objects).
            - A dictionary of matrices (keys: matrix names, values: numpy arrays).
        """
        return self.__load_all_scheme(self.load_graphs, self.load_matrices)

    def lazy_load(self) -> Optional[Tuple[pd.DataFrame, List[LiteralString], List[LiteralString]]]:
        """Load areas and list available graph and matrix filenames.

        Returns
        -------
        tuple[pandas.DataFrame, list[str], list[str]]
            A tuple containing:
            - A DataFrame with areas description (index: 'Id_Area').
            - A list of filenames (with '.json' extension) of all spatio-temporal graphs.
            - A list of filenames (with '.npy' extension) of all matrices.
        """
        return self.__load_all_scheme(lambda _: self.lazy_load_graphs(), self.lazy_load_matrices)

    def lazy_load_metrics(self) -> List[LiteralString]:
        return [filename for filename in self.__get_filenames('.csv') if filename.startswith('metrics_')]

    def load_metrics(self, name: str) -> Optional[pd.DataFrame]:
        with self.__within_archive as zfp:
            with zfp.open(f'metrics_{name}.csv', 'r') as fp:
                return load_metrics(fp)



type SavableDataElement = pd.DataFrame | SpatioTemporalGraphsDict | CorrelationMatricesDict


@dataclass
class DataSaver:
    """DataSaver accumulates and saves spatio-temporal data elements (areas, graphs, matrices) to a zip archive.

    Attributes
    ----------
    elements : list of SavableDataElement
        List of data elements to be saved. Each element can be a pandas DataFrame (areas),
        a dictionary of SpatioTemporalGraph objects, or a dictionary of correlation matrices.

    Methods
    -------
    add(element: SavableDataElement) -> None
        Add a data element to the list for saving.
    clear() -> None
        Clear all accumulated data elements.
    save(filepath: Path) -> None
        Save all accumulated data elements to the specified zip archive.
    """
    elements: List[SavableDataElement] = field(default_factory=lambda: [])

    def add(self, element: SavableDataElement) -> None:
        """Add a data element to the list for saving.

        Parameters
        ----------
        element : SavableDataElement
            The data element to add. Can be a pandas DataFrame, a dictionary of SpatioTemporalGraph objects,
            or a dictionary of correlation matrices.
        """
        self.elements.append(element)

    def clear(self) -> None:
        """Clear the list of all accumulated data elements for saving."""
        self.elements.clear()

    @staticmethod
    def __save_areas(areas: pd.DataFrame, zfp: ZipFile) -> None:
        with zfp.open('areas.csv', 'w') as fp:
            areas.to_csv(fp)

    @staticmethod
    def __save_graphs(graphs: SpatioTemporalGraphsDict, zfp: ZipFile):
        for name, graph in graphs.items():
            graph_dict = nx.json_graph.node_link_data(graph, edges='edges')
            graph_json = json.dumps(graph_dict, cls=_SpatioTemporalGraphEncoder)
            zfp.writestr(f'{name}.json', data=graph_json)

    @staticmethod
    def __save_matrices(matrices: CorrelationMatricesDict, zfp: ZipFile):
        for name, matrix in matrices.items():
            with zfp.open(f'{name}.npy', 'w') as fp:
                np.save(fp, matrix)

    def save(self, filepath: Path) -> None:
        """Save all accumulated data elements to the specified zip archive.

        Parameters
        ----------
        filepath : Path
            The path to the zip archive where the data will be saved.
        """
        with ZipFile(str(filepath), 'w') as zfp:
            for element in self.elements:
                if isinstance(element, pd.DataFrame):
                    self.__save_areas(element, zfp)
                elif isinstance(element, dict):
                    _, first = next(iter(element.items()))
                    if isinstance(first, SpatioTemporalGraph):
                        self.__save_graphs(element, zfp)
                    elif isinstance(first, np.ndarray):
                        self.__save_matrices(element, zfp)
