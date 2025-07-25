from typing import Any
from dataclasses import dataclass

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import split_factors_from_name
from ...graph import SpatioTemporalGraph
from ...io import DataLoader


@dataclass(frozen=True)
class GraphsDataset:
    # TODO docstring
    loader: DataLoader
    areas_desc: pd.DataFrame
    factors: list[set[str]]
    subjects: pd.DataFrame

    def serialize(self) -> dict[str, Any]:
        return {
            'filepath': str(self.loader.filepath),
            'areas_desc': self.areas_desc.reset_index().to_dict('records'),
            'factors': [list(f) for f in self.factors],
            'subjects': self.subjects.reset_index().to_dict('records')
        }

    def __contains__(self, ids: tuple[str, ...]) -> bool:
        return ids in self.subjects.index

    def get_graph(self, ids: tuple[str, ...]) -> SpatioTemporalGraph:
        filename = self.subjects.loc[ids]['Graph']
        return self.loader.load_graph(self.areas_desc, filename)

    def get_matrix(self, ids: tuple[str, ...]) -> np.ndarray:
        filename = self.subjects.loc[ids]['Matrix']
        return self.loader.load_matrix(filename)

    @staticmethod
    def deserialize(data: dict[str, Any]) -> 'GraphsDataset':
        filepath = data['filepath']
        n = len(data['factors'])
        subjects = pd.DataFrame(data['subjects'])
        return GraphsDataset(loader=DataLoader(filepath),
                             areas_desc=pd.DataFrame(data['areas_desc']).set_index('Id_Area'),
                             factors=[set(f) for f in data['factors']],
                             subjects=subjects.set_index(list(subjects.columns[:n+1])))

    @staticmethod
    def from_filepath(filepath: Path) -> 'GraphsDataset':
        # load the dataset lazily
        loader = DataLoader(filepath)
        result = loader.lazy_load()

        if result is None:
            raise IOError("No dataset red.")
        areas_desc, graphs_filenames, matrices_filenames = result

        # extract factors from filename (without extension
        filenames_without_ext = map(lambda n: n.split('.json')[0], graphs_filenames)
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
