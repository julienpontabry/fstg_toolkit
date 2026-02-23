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

import re
from pathlib import Path
from typing import Optional

from .utils import DockerLoader, DockerNotAvailableException, DockerImage


class SPMinerService:
    def __init__(self):
        try:
            self.__docker_loader = DockerLoader()
        except DockerNotAvailableException as e:
            raise RuntimeError("Unable to initialize SPMiner service.") from e

        self.__docker_image: Optional[DockerImage] = None
        self.__progress_reg = re.compile(r'^\[(\d+)/(\d+)]')

    def prepare(self):
        if self.__docker_image is None:
            # TODO use an external config file?
            tag = 'spminer:latest'
            build_path = Path(__file__).parent.parent / 'spminer'
            self.__docker_image = self.__docker_loader.load_local_image(tag, build_path)

    def run(self, input_dir: Path, output_dir: Path):
        self.prepare()  # makes sure docker image is set

        output = self.__docker_image.run(
            volumes={str(input_dir.resolve()): {'bind': '/app/data', 'mode': 'ro'},
                     str(output_dir.resolve()): {'bind': '/app/results_batch', 'mode': 'rw'}},
            stdout=True,
            stderr=True
        )

        for line in output:
            if len(line) < 10:
                if match := self.__progress_reg.match(line):
                    # FIXME why does it slow down/block the output after the first one?
                    print(int(match.group(1)), match.group(2))
            # print(line, end='', flush=True)

# TODO add frequent patterns classes
