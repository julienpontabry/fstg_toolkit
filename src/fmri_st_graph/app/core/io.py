from typing import Any
from dataclasses import dataclass

from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from .utils import split_factors_from_name


@dataclass(frozen=True)
class GraphsDataset:
    filepath: Path
    areas_desc: pd.DataFrame
    subjects: pd.DataFrame

    def serialize(self) -> dict[str, Any]:
        return {
            'filepath': str(self.filepath),
            'areas_desc': self.areas_desc.reset_index().to_dict('records'),
            'subjects': self.subjects.reset_index().to_dict('records')
        }

    @staticmethod
    def deserialize(data: dict[str, Any]) -> 'GraphsDataset':
        return GraphsDataset(filepath=data['filepath'],
                             areas_desc=data['areas_desc'],
                             subjects=data['subjects'])

    @staticmethod
    def from_filepath(filepath: Path) -> 'GraphsDataset':
        with ZipFile(str(filepath), 'r') as zfp:
            # read the areas description from csv file
            with zfp.open('areas.csv', 'r') as fp:
                areas_desc = pd.read_csv(fp, index_col='Id_Area')

            # build the subjects from the name of the included graphs
            filenames = [name.split('.json')[0] for name in zfp.namelist()
                         if name.endswith('.json')]
            factors, ids = split_factors_from_name(filenames)
            subjects = pd.DataFrame({'Subject': filenames})

        return GraphsDataset(filepath=filepath,
                             areas_desc=areas_desc,
                             subjects=subjects)

    def __str__(self) -> str:
        return f"GraphsDataset(filepath=\"{self.filepath}\", "\
               f"#areas={len(self.areas_desc)}, #subjects={len(self.subjects)})"

    def __repr__(self) -> str:
        return str(self)
