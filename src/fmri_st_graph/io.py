"""Defines helpers for inputs/outputs."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Dict, LiteralString
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


# @deprecated("Use the data loader instead")
# TODO remove usage
def load_spatio_temporal_graph(filepath: Path | str) -> SpatioTemporalGraph:
    """Load a spatio-temporal graph from its zip file.

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
    See {`zipfile`} for exceptions and errors raised.
    """
    with ZipFile(str(filepath), 'r') as zfp:
        # read the graph from json file
        with zfp.open('graph.json', 'r') as fp:
            graph_dict = json.load(fp, object_hook=_spatio_temporal_object_hook)
            graph = nx.json_graph.node_link_graph(graph_dict, edges='edges')

        # read the areas description from csv file
        with zfp.open('areas.csv', 'r') as fp:
            areas = pd.read_csv(fp, index_col='Id_Area')

        return SpatioTemporalGraph(graph, areas)


# @deprecated("Use the data loader instead")
# TODO remove usage
def load_spatio_temporal_graphs(filepath: Path | str) -> dict[str, SpatioTemporalGraph]:
    """Load multiple spatio-temporal graphs from a zip file.

    Parameters
    ----------
    filepath: Path | str
        The path to the zip file.

    Returns
    -------
    dict[str, SpatioTemporalGraph]
        A dictionary of spatio-temporal graphs contained in the zip file, where the keys are
        the names of the graphs (the filenames without extension).

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
    ...     'Name_Region': ['R1', 'R2', 'R3']}, index=[1, 2, 3])
    >>> graph_path = Path('/tmp/tmp.zip')
    >>> graph_struct1 = SpatioTemporalGraph(G, areas_desc)
    >>> graph_struct2 = SpatioTemporalGraph(G, areas_desc)
    >>> save_spatio_temporal_graphs({'g1': graph_struct1, 'g2': graph_struct2}, graph_path)
    >>> graphs_dict = load_spatio_temporal_graphs(graph_path)

    Raises
    ------
    See {`zipfile`} for exceptions and errors raised.
    """
    graphs = {}

    with ZipFile(str(filepath), 'r') as zfp:
        # read the areas description from csv file
        with zfp.open('areas.csv', 'r') as fp:
            areas = pd.read_csv(fp, index_col='Id_Area')

        # read the graphs from the json files
        for name in zfp.namelist():
            if name.endswith('.json'):
                with zfp.open(name, 'r') as fp:
                    graph_dict = json.load(fp, object_hook=_spatio_temporal_object_hook)
                    graph = nx.json_graph.node_link_graph(graph_dict, edges='edges')
                    graphs[name.split('.json')[0]] = SpatioTemporalGraph(graph, areas)

    return graphs


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
    with ZipFile(str(filepath), 'w') as zfp:
        # write the graph into json file
        graph_dict = nx.json_graph.node_link_data(graph, edges='edges')
        graph_json = json.dumps(graph_dict, indent=4, cls=_SpatioTemporalGraphEncoder)
        zfp.writestr('graph.json', data=graph_json)  # json cannot dump to a binary file pointer

        # write the areas description into csv file
        with zfp.open('areas.csv', 'w') as fp:
            graph.areas.to_csv(fp)

# TODO refactor module to include optional elements in the archive (eg matrices)
def save_spatio_temporal_graphs(graphs: dict[str, SpatioTemporalGraph], filepath: Path | str) -> None:
    """Save multiple spatio-temporal graphs to a zip file.

    Note that no check is done to ensure the graphs share the same areas description.

    Parameters
    ----------
    graphs: dict[str, SpatioTemporalGraph]
        The spatio-temporal graphs to save.
    filepath: Path | str
        The path to the zip file.

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
    >>> graph_struct1 = SpatioTemporalGraph(G, areas_desc)
    >>> graph_struct2 = SpatioTemporalGraph(G, areas_desc)
    >>> save_spatio_temporal_graphs({'g1': graph_struct1, 'g2': graph_struct2}, graph_path)
    """
    with ZipFile(str(filepath), 'w') as zfp:
        # write the graphs into json file
        for name, graph in graphs.items():
            graph_dict = nx.json_graph.node_link_data(graph, edges='edges')
            graph_json = json.dumps(graph_dict, indent=4, cls=_SpatioTemporalGraphEncoder)
            zfp.writestr(f'{name}.json', data=graph_json)

        # write the areas description into csv file
        if len(graphs) > 0:
            areas = next(iter(graphs.values())).areas
            with zfp.open('areas.csv', 'w') as fp:
                areas.to_csv(fp)


type SpatioTemporalGraphsDict = Dict[LiteralString, SpatioTemporalGraph]
type SpatioTemporalGraphsFilenamesList = List[LiteralString]
type AnySpatioTemporalGraphsStruct = SpatioTemporalGraphsDict | SpatioTemporalGraphsFilenamesList
type GraphsLoader = Callable[[pd.DataFrame], AnySpatioTemporalGraphsStruct]
type CorrelationMatricesDict = Dict[LiteralString, np.ndarray]


@dataclass
class DataLoader:
    # TODO docstring
    filepath: Path

    @property
    def __within_archive(self):
        return ZipFile(str(self.filepath), 'r')

    def load_areas(self) -> Optional[pd.DataFrame]:
        with self.__within_archive as zfp:
            with zfp.open('areas.csv', 'r') as fp:
                return pd.read_csv(fp, index_col='Id_Area')

    def load_graphs(self, areas: pd.DataFrame) -> SpatioTemporalGraphsDict:
        graphs = {}

        with self.__within_archive as zfp:
            for name in zfp.namelist():
                if name.endswith('.json'):
                    with zfp.open(name, 'r') as fp:
                        graph_dict = json.load(fp, object_hook=_spatio_temporal_object_hook)
                        graph = nx.json_graph.node_link_graph(graph_dict, edges='edges')
                        graphs[name.split('.json')[0]] = SpatioTemporalGraph(graph, areas)

        return graphs

    def lazy_load_graphs(self) -> SpatioTemporalGraphsFilenamesList:
        with self.__within_archive as zfp:
            return list(filter(lambda n: n.endswith('.json'), zfp.namelist()))

    def load_graph(self, areas: pd.DataFrame, filename: LiteralString) -> Optional[SpatioTemporalGraph]:
        with self.__within_archive as zfp:
            with zfp.open(filename, 'r') as fp:
                graph_dict = json.load(fp, object_hook=_spatio_temporal_object_hook)
                graph = nx.json_graph.node_link_graph(graph_dict, edges='edges')
                return SpatioTemporalGraph(graph, areas)

    def __load_all_scheme(self, loader: GraphsLoader) -> Optional[Tuple[pd.DataFrame, AnySpatioTemporalGraphsStruct]]:
        areas = self.load_areas()

        if areas is None:
            return None

        graphs = loader(areas)
        return areas, graphs

    def load(self) -> Optional[Tuple[pd.DataFrame, SpatioTemporalGraphsDict]]:
        return self.__load_all_scheme(self.load_graphs)

    def lazy_load(self) -> Optional[Tuple[pd.DataFrame, SpatioTemporalGraphsFilenamesList]]:
        return self.__load_all_scheme(lambda a: self.lazy_load_graphs())


type SavableDataElement = pd.DataFrame | SpatioTemporalGraphsDict | CorrelationMatricesDict


@dataclass
class DataSaver:
    # TODO docstring
    elements: List[SavableDataElement] = field(default_factory=lambda: [])

    def add(self, element: SavableDataElement) -> None:
        self.elements.append(element)

    def clear(self) -> None:
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
        for name, matrix in matrices:
            with zfp.open(f'{name}.npy', 'w') as fp:
                np.save(fp, matrix)

    def save(self, filepath: Path) -> None:
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
