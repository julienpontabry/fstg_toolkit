"""Defines spatio-temporal graphs and related structures."""
from enum import Enum, auto, unique

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
        The conditions on the nodes of the subgraph as keywords arguments.

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
    """
    return graph.subgraph([node
                           for node, data in graph.nodes.items()
                           if all([data[k] == v for k, v in conditions.items()])])


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
