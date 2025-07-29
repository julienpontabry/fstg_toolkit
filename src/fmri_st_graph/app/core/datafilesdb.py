from typing import Optional, Generator, Type
from pathlib import Path
from abc import ABC, abstractmethod

import secrets


class DataFilesDB(ABC):
    """Abstract base class for managing a database of data files, each associated with a unique token.

    Parameters
    ----------
    token_nb_bytes : int, optional
        Number of bytes to use when generating unique tokens for data files (default is 3).

    Methods
    -------
    add(file_path: Path) -> str
        Adds a data file to the database and returns its unique token.
    get(token: str) -> Optional[pathlib.Path]
        Retrieves the file path associated with the given token.
    list() -> Generator[tuple[str, pathlib.Path], None, None]
        Lists all token-file path pairs in the database.
    """

    def __init__(self, token_nb_bytes: int = 3):
        self.__token_nb_bytes = token_nb_bytes

    @abstractmethod
    def _add_data_file_to_db(self, token: str, file_path: Path) -> None:
        raise NotImplementedError("Abstract class not meant to be used directly")

    def __generate_token(self) -> str:
        return secrets.token_urlsafe(nbytes=self.__token_nb_bytes)

    def add(self, file_path: Path) -> str:
        """Adds a data file to the database and returns its unique token.

        Parameters
        ----------
        file_path : Path
            The path to the data file to be added.

        Returns
        -------
        str
            A unique token associated with the added data file.
        """
        token = self.__generate_token()
        self._add_data_file_to_db(token, file_path)
        return token

    @abstractmethod
    def get(self, token: str) -> Optional[Path]:
        """Retrieves the file path associated with the given token.

        Parameters
        ----------
        token : str
            The unique token associated with the data file.

        Returns
        -------
        pathlib.Path or None
            The path to the data file if found, otherwise None.
        """
        raise NotImplementedError("Abstract class not meant to be used directly")

    @abstractmethod
    def list(self) -> Generator[tuple[str, Path], None, None]:
        """Lists all token-file path pairs in the database.

        Returns
        -------
        Generator of tuple[str, pathlib.Path]
            A generator yielding tuples of tokens and their associated file paths.
        """
        raise NotImplementedError("Abstract class not meant to be used directly")

    def __iter__(self):
        return iter(self.list())

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError("Abstract class not meant to be used directly")

    def __str__(self) -> str:
        return f"token_nb_bytes={self.__token_nb_bytes}"


class MemoryDataFilesDB(DataFilesDB):
    """In-memory implementation of the DataFilesDB abstract base class.

    This class manages a database of data files using a Python dictionary,
    mapping unique tokens to file paths. It is suitable for use cases where
    persistence is not required and the number of files is relatively small.

    Parameters
    ----------
    token_nb_bytes : int, optional
        Number of bytes to use when generating unique tokens for data files (default is 3).

    Methods
    -------
    add(file_path: Path) -> str
        Adds a data file to the database and returns its unique token.
    get(token: str) -> Optional[Path]
        Retrieves the file path associated with the given token.
    list() -> Generator[tuple[str, Path], None, None]
        Lists all token-file path pairs in the database.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__db: dict[str, Path] = {}  # DB is represented by a dictionary mapping a token to a data path's file

    def _add_data_file_to_db(self, token: str, file_path: Path) -> None:
        self.__db[token] = file_path

    def get(self, token: str) -> Optional[Path]:
        return self.__db.get(token)

    def list(self) -> Generator[tuple[str, Path], None, None]:
        for token in self.__db:
            yield token, self.__db[token]

    def __len__(self) -> int:
        return len(self.__db)

    def __str__(self) -> str:
        return f"MemoryDataFilesDB({super().__str__()})"

    def __repr__(self) -> str:
        return str(self)


class SQLiteDataFilesDB(DataFilesDB):
    pass  # TODO implement the sqlite backend for remote serving (larger number of files with persistence)


singleton_data_files_db: Optional[DataFilesDB] = None


def get_data_file_db(requested_type: Optional[Type[DataFilesDB]] = None) -> DataFilesDB:
    """Returns the singleton instance of the data files database.

    If the singleton instance does not exist, it is created using the specified type
    or defaults to SQLiteDataFilesDB.

    Parameters
    ----------
    requested_type : type[DataFilesDB] or None, optional
        The class type of the data files database to instantiate. If None, defaults to
        SQLiteDataFilesDB for database creation.

    Returns
    -------
    DataFilesDB
        The singleton instance of the data files database.

    Raises
    ------
    RuntimeError
        If a database instance already exists with a different type than requested_type.
    """
    global singleton_data_files_db

    if singleton_data_files_db is None:
        db_type = requested_type or SQLiteDataFilesDB
        singleton_data_files_db = db_type()
    elif requested_type is not None and not isinstance(singleton_data_files_db, requested_type):
        raise RuntimeError(f"Unable to get a data file db of type {requested_type}! "
                           f"A DB already exists with type {type(singleton_data_files_db)}")

    return singleton_data_files_db
