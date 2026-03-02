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
