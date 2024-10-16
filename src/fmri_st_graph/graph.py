"""Defines spatio-temporal graphs and related structures."""

from enum import Enum, auto, unique
from dataclasses import dataclass

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
        """
        Find the RC5 transition from its name.

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


@dataclass
class SpatioTemporalGraph:
    """Defines a spatio-temporal graph for functional connectivity data."""
    graph: nx.DiGraph
    areas: pd.DataFrame
