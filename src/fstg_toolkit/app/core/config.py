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
from typing import Any, Optional


class NotConfiguredError(Exception):
    """Raised when a required configuration attribute has not been set."""

    def __init__(self, name: str):
        """Initialise with the name of the missing configuration attribute.

        Parameters
        ----------
        name: str
            The name of the attribute that is not configured.
        """
        super().__init__(f"The element {name} is not configured!")


@dataclass
class __Config:
    """Singleton application configuration container.

    Attributes are set to ``None`` by default and must be configured before use.
    Accessing an unconfigured attribute raises :class:`NotConfiguredError`.

    Attributes
    ----------
    db_path: Path or None
        Path to the SQLite database file used by the dashboard.
    data_path: Path or None
        Path to the directory where processed dataset archives are stored.
    upload_path: Path or None
        Path to the temporary upload directory.
    max_processing_queue_workers: int
        Maximum number of concurrent dataset processing jobs (default 1).
    max_processing_cpus: int
        Maximum number of CPUs allocated per processing job (default 4).
    """

    db_path: Optional[Path] = None
    data_path: Optional[Path] = None
    upload_path: Optional[Path] = None
    max_processing_queue_workers: int = 1
    max_processing_cpus: int = 4

    def is_configured(self, name) -> bool:
        """Check whether a configuration attribute has been set.

        Parameters
        ----------
        name: str
            The attribute name to check.

        Returns
        -------
        bool
            True if the attribute is not ``None``; False otherwise.
        """
        return super().__getattribute__(name) is not None

    def __getattribute__(self, name: str) -> Any:
        """Return the attribute value, raising :class:`NotConfiguredError` if unset.

        Parameters
        ----------
        name: str
            The attribute name to retrieve.

        Raises
        ------
        NotConfiguredError
            If the attribute exists but has not been configured (is ``None``).
        """
        if value := super().__getattribute__(name):
            return value
        else:
            raise NotConfiguredError(name)

    def __repr__(self) -> str:
        """Return a human-readable representation of the current configuration."""
        fields = []
        for field in self.__dataclass_fields__:
            try:
                value = object.__getattribute__(self, field)
                fields.append(f"{field}={value}")
            except AttributeError:
                fields.append(f"{field}=None")
        return f"Config({', '.join(fields)})"



config = __Config()
