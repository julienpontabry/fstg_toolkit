from typing import Any, List, Dict, Set, Tuple, Optional
from dataclasses import dataclass

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import split_factors_from_name
from ...graph import SpatioTemporalGraph
from ...io import DataLoader


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
    factors: List[Set[str]]
    subjects: pd.DataFrame

    def serialize(self) -> Dict[str, Any]:
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

    def __contains__(self, ids: Tuple[str, ...]) -> bool:
        return ids in self.subjects.index

    def get_graph(self, ids: Tuple[str, ...]) -> SpatioTemporalGraph:
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

    def get_matrix(self, ids: Tuple[str, ...]) -> np.ndarray:
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

    def has_metrics(self) -> bool:
        return self.loader.load_metrics() is not None

    def get_metrics(self) -> Optional[pd.DataFrame]:
        return self.loader.load_metrics()

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> 'GraphsDataset':
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
        filepath = data['filepath']
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
        result = loader.lazy_load()

        if result is None:
            raise IOError("No dataset red.")
        areas_desc: pd.DataFrame
        areas_desc, graphs_filenames, matrices_filenames = result

        # extract factors from filename (without extension
        filenames_without_ext = map(lambda name: name.split('.json')[0], graphs_filenames)
        factors: List[Set[str]]
        factors, ids = split_factors_from_name(filenames_without_ext)

        # create a subject's table with factors as index and filenames as data
        data = list(zip(*zip(*ids), graphs_filenames))
        n = len(factors)
        columns = [f'Factor{i}' for i in range(n)] + ['Subject', 'Graph']
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
