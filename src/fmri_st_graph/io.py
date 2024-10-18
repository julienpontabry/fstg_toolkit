"""Defines helpers for inputs/outputs."""

from pathlib import Path
from zipfile import ZipFile
import json

import pandas as pd
import networkx as nx

from .graph import RC5, SpatioTemporalGraph


class __SpatioTemporalGraphEncoder(json.JSONEncoder):
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


def __spatio_temporal_object_hook(obj: dict) -> dict:
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


def load_spatio_temporal_graph(filepath: Path) -> SpatioTemporalGraph:
    """Load a spatio-temporal graph from its zip file.

    Parameters
    ----------
    filepath: Path
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
    ...     'Name_Area': ['Area 1', 'Area 2', 'Area 3'],
    ...     'Name_Region': ['R1', 'R2', 'R3']}, index=[1, 2, 3])
    >>> graph_path = Path('/tmp/tmp.zip')
    >>> save_spatio_temporal_graph(SpatioTemporalGraph(G, areas_desc), graph_path)
    >>> graph_struct = load_spatio_temporal_graph(graph_path)
    """
    with ZipFile(str(filepath), 'r') as zfp:
        # read the graph from json file
        with zfp.open('graph.json', 'r') as fp:
            graph_dict = json.load(fp, object_hook=__spatio_temporal_object_hook)
            graph = nx.json_graph.node_link_graph(graph_dict, edges='edges')

        # read the areas description from csv file
        with zfp.open('areas.csv', 'r') as fp:
            areas = pd.read_csv(fp)

        return SpatioTemporalGraph(graph, areas)


def save_spatio_temporal_graph(graph: SpatioTemporalGraph, filepath: Path) -> None:
    """Save a spatio-temporal graph to a zip file.

    Parameters
    ----------
    graph: SpatioTemporalGraph
        The spatio-temporal graph to save.
    filepath: Path
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
        graph_dict = nx.json_graph.node_link_data(graph.graph, edges='edges')
        graph_json = json.dumps(graph_dict, indent=4, cls=__SpatioTemporalGraphEncoder)
        zfp.writestr('graph.json', data=graph_json)  # json cannot dump to a binary file pointer

        # write the areas description into csv file
        with zfp.open('areas.csv', 'w') as fp:
            graph.areas.to_csv(fp)