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
        """
        available = [e for e in RC5 if e.name == name]

        if len(available) > 0:
            return available[0]
        else:
            raise ValueError(f"Unable to find a transition named \"{name}\"!")


@dataclass
class SpatioTemporalGraph:
    graph: nx.DiGraph
    areas: pd.DataFrame
