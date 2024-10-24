"""Defines spatio-temporal graphs and related structures."""
from collections.abc import Iterable
from enum import Enum, auto, unique
from math import isclose
from numbers import Number

import networkx as nx
import pandas as pd


@unique
class RC5(Enum):
    """Defines an RC5 temporal transition."""
    EQ = auto()
    PP = auto()
    PPi = auto()
    PO = auto()
    DC = auto()

    @staticmethod
    def from_name(name: str) -> 'RC5':
        """Find the RC5 transition from its name.

        Parameters
        ----------
        name: string
            The name of the RC5 transition.

        Returns
        -------
        RC5
            The corresponding RC5 transition if it has been found.

        Raises
        ------
        ValueError: if no RC5 transition can be found with the given name.

        Examples
        --------
        Access to a transition is simple as typing the right transition.
        >>> RC5.PPi
        <RC5.PPi: 3>

        To access to all the available transition, one can iterate on the RC5 enumeration.
        >>> for transition in RC5:
        ...     print(transition)
        ...
        RC5.EQ
        RC5.PP
        RC5.PPi
        RC5.PO
        RC5.DC

        When only the transition name is available, the transition itself can be retrieved
        from it with the `from_name` static method.
        >>> RC5.from_name("PO")
        <RC5.PO: 4>
        >>> RC5.from_name("EQ")
        <RC5.EQ: 1>

        When the provided name does not match any available transition, a `ValueError`
        transition is thrown.
        >>> RC5.from_name("NN")
        Traceback (most recent call last):
        ValueError: Unable to find a transition named "NN"!
        """
        available = [e for e in RC5 if e.name == name]

        if len(available) > 0:
            return available[0]
        else:
            raise ValueError(f"Unable to find a transition named \"{name}\"!")


def subgraph(graph: nx.Graph, **conditions: any) -> nx.Graph:
    """Take the subgraph that matches the conditions on the nodes.

    Parameters
    ----------
    graph: nx.Graph
        The initial graph.
    conditions: dict[str, any]
        The conditions on the nodes of the subgraph as keywords arguments. As value,
        any single value or iterable of values is supported.

    Returns
    -------
    nx.Graph
        The subgraph matching the specified conditions.

    Example
    -------
    >>> G = nx.Graph()
    >>> G.add_nodes_from([(1, dict(a=0, b=1)), (2, dict(a=2, b=1)), (3, dict(a=2, b=2)), (4, dict(a=2, b=1))])
    >>> G.add_edges_from([(1, 2), (3, 4), (1, 4)])
    >>> subgraph(G).nodes
    NodeView((1, 2, 3, 4))
    >>> subgraph(G, a=0).nodes
    NodeView((1,))
    >>> subgraph(G, b=1).nodes
    NodeView((1, 2, 4))
    >>> subgraph(G, a=2).nodes
    NodeView((2, 3, 4))
    >>> subgraph(G, a=2, b=2).nodes
    NodeView((3,))
    >>> subgraph(G, b=(1, 2), a=2).nodes
    NodeView((2, 3, 4))
    >>> subgraph(G, b=range(1, 3)).nodes
    NodeView((1, 2, 3, 4))
    """
    def __process(d: any, v: any):
        if isinstance(v, Iterable):
            return d in v
        else:
            return d == v

    return graph.subgraph([node
                           for node, data in graph.nodes.items()
                           if all([__process(data[k], v) for k, v in conditions.items()])])


class SpatioTemporalGraph(nx.DiGraph):
    def __init__(self, graph: nx.DiGraph = None, areas: pd.DataFrame = None) -> None:
        super().__init__(graph)
        self.areas = areas

    @property
    def time_range(self) -> range:
        """Get the time range covered by the spatio-temporal graph."""
        return range(self.graph['max_time']+1)

    def conditional_subgraph(self, **conditions) -> 'SpatioTemporalGraph':
        """Helper to take the subgraph of the spatio-temporal graph matching the specified conditions.

        See :func:`~graph.subgraph` for the arguments.
        """
        return SpatioTemporalGraph(subgraph(self, **conditions), self.areas)

    def __eq__(self, other: 'SpatioTemporalGraph') -> 'SpatioTemporalGraph':
        return nx.utils.graphs_equal(self, other) and self.areas.equals(other.areas)


def __data_almost_equal(data1: dict[str, any], data2: dict[str, any],
                        rel_tol: float = 1e-9, abs_tol: float = 0.0):
    if len(data1) != len(data2):
        return False

    for key, item in data1.items():
        if key not in data2:
            return False
        elif isinstance(item, Number):
            if not isclose(data1[key], data2[key],
                           rel_tol=rel_tol, abs_tol=abs_tol):
                return False
        elif data1[key] != data2[key]:
            return False

    return True


def are_st_graphs_close(graph1: SpatioTemporalGraph, graph2: SpatioTemporalGraph) -> bool:
    """Test if two spatio-temporal graphs are equal with some tolerance on numerical values.

    Parameters
    ----------
    graph1: SpatioTemporalGraph
        The first spatio-temporal graph to compare.
    graph2: SpatioTemporalGraph
        The second spatio-temporal graph to compare.

    Returns
    -------
    bool
        True if graphs are almost equal; false otherwise.

    Examples
    --------
    >>> g1 = nx.DiGraph()
    >>> g1.add_nodes_from([(1, dict(t=0, areas={1, 2}, region="R1", internal_strength=0.8)),
    ...                    (2, dict(t=0, areas={3}, region="R1", internal_strength=1)),
    ...                    (3, dict(t=1, areas={1, 2, 3}, region="R1", internal_strength=0.9))])
    >>> g1.add_edges_from([(1, 3, dict(transition=RC5.PP, type='temporal'))])
    >>> a1 = pd.DataFrame({'Id_Area': [1, 2, 3], 'Name_Area': ["A1", "A2", "A3"], 'Name_Region': ["R1", "R1", "R1"]})
    >>> st_g1 = SpatioTemporalGraph(g1, a1)
    >>> g2 = nx.DiGraph()
    >>> g2.add_nodes_from([(1, dict(t=0, areas={1, 2}, region="R1", internal_strength=0.7999999999999)),
    ...                    (2, dict(t=0, areas={3}, region="R1", internal_strength=1.000000000001)),
    ...                    (3, dict(t=1, areas={1, 2, 3}, region="R1", internal_strength=0.8999999999999))])
    >>> g2.add_edges_from([(1, 3, dict(transition=RC5.PP, type='temporal'))])
    >>> a2 = pd.DataFrame({'Id_Area': [1, 2, 3], 'Name_Area': ["A1", "A2", "A3"], 'Name_Region': ["R1", "R1", "R1"]})
    >>> st_g2 = SpatioTemporalGraph(g2, a2)
    >>> g3 = nx.DiGraph()
    >>> g3.add_nodes_from([(1, dict(t=0, areas={1, 2}, region="R1", internal_strength=0.8)),
    ...                    (2, dict(t=1, areas={1, 2}, region="R1", internal_strength=0.8))])
    >>> g3.add_edges_from([(1, 2, dict(transition=RC5.EQ, type='temporal'))])
    >>> a3 = pd.DataFrame({'Id_Area': [1, 2], 'Name_Area': ["A1", "A2"], 'Name_Region': ["R1", "R1"]})
    >>> st_g3 = SpatioTemporalGraph(g3, a3)
    >>> are_st_graphs_close(st_g1, st_g1)
    True
    >>> are_st_graphs_close(st_g1, st_g2)
    True
    >>> are_st_graphs_close(st_g1, st_g3)
    False
    """
    nodes1 = graph1.nodes
    nodes2 = graph2.nodes
    nodes_equal = list(nodes1) == list(nodes2)
    if not nodes_equal:
        return False
    nodes_data_almost_equal = all(__data_almost_equal(nodes1[n], nodes2[n])
                                  for n in nodes1)

    edges1 = graph1.edges
    edges2 = graph2.edges
    edges_equal = list(edges1) == list(edges2)
    if not edges_equal:
        return False
    edges_data_almost_equal = all(__data_almost_equal(edges1[e], edges2[e])
                                  for e in edges1)

    return nodes_equal and edges_equal and nodes_data_almost_equal and \
        edges_data_almost_equal and graph1.areas.equals(graph2.areas)
