# Copyright 2025 ICube (University of Strasbourg - CNRS)
# author: Julien PONTABRY (ICube)
#
# This software is a computer program whose purpose is to provide a toolkit
# to model, process and analyze the longitudinal reorganization of brain
# connectivity data, as functional MRI for instance.
#
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/or redistribute the software under the terms of the CeCILL-B
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL-B license and that you accept its terms.

from dataclasses import dataclass
from pathlib import Path


class InvalidSubmittedDataset(Exception):
    def __init__(self, message: str):
        super().__init__(message)


@dataclass(frozen=True)
class SubmittedDataset:
    name: str
    include_raw: bool
    compute_metrics: bool
    areas_file: Path
    matrices_files: list[Path]

    def __post_init__(self):
        if self.name == "":
            raise InvalidSubmittedDataset("The dataset's name must be non-empty!")

        if not self.areas_file:
            raise InvalidSubmittedDataset("The areas description file must be non-empty!")

        if not self.areas_file.exists() or not self.areas_file.is_file():
            raise InvalidSubmittedDataset("The areas description file does not exist!")

        if not self.matrices_files:
            raise InvalidSubmittedDataset("There must be at least one matrices file!")

        if any(not Path(f).exists() for f in self.matrices_files):
            raise InvalidSubmittedDataset("The matrices files must all exist!")
