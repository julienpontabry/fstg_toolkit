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

import docker


@dataclass(frozen=True)
class DockerImage:
    image: docker.models.images.Image


@dataclass(frozen=True)
class DockerClient:
    client: docker.DockerClient = docker.from_env()

    def is_available(self) -> bool:
        try:
            self.client.ping()
            return True
        except docker.errors.DockerException:
            return False

    def load_image(self, tag: str, path: Path) -> DockerImage:
        try:
            return DockerImage(self.client.images.get(tag))
        except docker.errors.ImageNotFound:
            # TODO how to display on CLI that it is building (spinner display)
            image, logs = self.client.images.build(path=str(path), tag=tag)
            for chunk in logs:
                if 'stream' in chunk:
                    print(chunk['stream'], end='', flush=True)
            return DockerImage(image)
