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
from typing import Generator

import docker


class DockerException(Exception):
    def __init__(self, message: str):
        super().__init__(message)


class DockerImageException(DockerException):
    def __init__(self, message: str):
        super().__init__(message)


@dataclass(frozen=True)
class DockerImage:
    client: docker.DockerClient
    image_tag: str

    def run(self, **kwargs) -> Generator[str, None, None]:
        if 'command' in kwargs and len(kwargs['command']) > 0:
            kwargs['command'] = kwargs['command'].split(' ')

        try:
            container = self.client.containers.create(image=self.image_tag, **kwargs)
            container.start()
            for chunk in container.logs(stream=True, stderr=True, stdout=True, follow=True):
                yield chunk.decode()
        except docker.errors.ContainerError as e:
            raise DockerImageException("Container exited with non-zero code.") from e
        except docker.errors.ImageNotFound as e:
            raise DockerImageException("Image not found.") from e
        except docker.errors.APIError as e:
            raise DockerException("Docker server returned an error.") from e


class DockerNotAvailableException(DockerException):
    def __init__(self):
        super().__init__("Docker is not available.")


class DockerClient:
    def __init__(self):
        try:
            self.client = docker.from_env()
            self.client.ping()
        except docker.errors.DockerException as e:
            raise DockerNotAvailableException() from e

    def load_local_image(self, tag: str, path: Path) -> DockerImage:
        try:
            _ = self.client.images.get(tag)
            return DockerImage(self.client, tag)
        except docker.errors.ImageNotFound:
            _, logs = self.client.images.build(path=str(path), tag=tag)
            for chunk in logs:
                if 'stream' in chunk:
                    print(chunk['stream'], end='', flush=True)
            return DockerImage(self, tag)
