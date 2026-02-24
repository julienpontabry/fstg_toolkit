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

"""Utility module for the FSTG toolkit.

This module provides helper classes and functions for interacting with Docker containers,
including loading Docker images, running containers, and handling Docker-related
errors. It is designed to support the FSTG toolkit's functionality that requires
Docker integration.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Any

import docker


class DockerException(Exception):
    """Base exception class for Docker-related errors.
    
    Parameters
    ----------
    message : str
        The error message describing the Docker-related issue.
    """

    def __init__(self, message: str):
        super().__init__(message)


class DockerImageException(DockerException):
    """Exception class for Docker image-related errors.
    
    Parameters
    ----------
    message : str
        The error message describing the Docker image-related issue.
    """

    def __init__(self, message: str):
        super().__init__(message)


@dataclass(frozen=True)
class DockerImage:
    """A class representing a Docker image with methods to run the image into containers.
    
    Attributes
    ----------
    tag : str
        The tag/name of the Docker image.
    """
    __client: docker.DockerClient
    tag: str

    @staticmethod
    def __get_rm_or_default(kwargs: dict, key: str, default: Any = None) -> Any:
        """Helper method to get and remove a key from kwargs dictionary.
        
        Parameters
        ----------
        kwargs : dict
            The dictionary to search in.
        key : str
            The key to look for.
        default : Any, optional
            Default value to return if key is not found. Defaults to None.
            
        Returns
        -------
        Any
            The value associated with the key, or the default value if key not found.
        """
        if key in kwargs:
            output = kwargs[key]
            del kwargs[key]
            return output
        else:
            return default

    def run(self, **kwargs) -> Generator[str, None, None]:
        """Run a container from this Docker image.
        
        Parameters
        ----------
        **kwargs
            Additional arguments to pass to container creation.
            
        Yields
        ------
        str
            Output chunks from the container logs.
            
        Raises
        ------
        DockerImageException
            If container exits with non-zero code or image is not found.
        DockerException
            If Docker server returns an error.
        """
        if 'command' in kwargs and len(kwargs['command']) > 0:
            kwargs['command'] = kwargs['command'].split(' ')

        stdout = self.__get_rm_or_default(kwargs, 'stdout', True)
        stderr = self.__get_rm_or_default(kwargs, 'stderr', True)

        try:
            container = self.__client.containers.create(
                image=self.tag,
                user=f'{os.getuid()}:{os.getgid()}',
                **kwargs)
            container.start()
            for chunk in container.logs(stream=True, stderr=stderr, stdout=stdout, follow=True):
                yield chunk.decode()
            container.remove()
        except docker.errors.ContainerError as e:
            raise DockerImageException("Container exited with non-zero code.") from e
        except docker.errors.ImageNotFound as e:
            raise DockerImageException("Image not found.") from e
        except docker.errors.APIError as e:
            raise DockerException("Docker server returned an error.") from e


class DockerNotAvailableException(DockerException):
    """Exception raised when Docker is not available or cannot be accessed."""

    def __init__(self):
        super().__init__("Docker is not available.")


class DockerHelper:
    """A class for loading Docker images, either from local cache or by building them."""

    def __init__(self):
        """Initialize the DockerHelper by creating a Docker client and testing the connection.
        
        Raises
        ------
        DockerNotAvailableException
            If Docker is not available or cannot be accessed.
        """
        try:
            self.__client = docker.from_env()
            self.__client.ping()
        except docker.errors.DockerException as e:
            raise DockerNotAvailableException() from e

    def load_local_image(self, tag: str, path: Path) -> DockerImage:
        """Load a Docker image, either from cache or by building it from a local path.
        
        Parameters
        ----------
        tag : str
            The tag to assign to the Docker image.
        path : Path
            The path to the directory containing the Dockerfile.
            
        Returns
        -------
        DockerImage
            An instance of DockerImage representing the loaded image.
        """
        try:
            _ = self.__client.images.get(tag)
            return DockerImage(self.__client, tag)
        except docker.errors.ImageNotFound:
            _, logs = self.__client.images.build(path=str(path), tag=tag)
            for chunk in logs:
                if 'stream' in chunk:
                    print(chunk['stream'], end='', flush=True)
            return DockerImage(self.__client, tag)
