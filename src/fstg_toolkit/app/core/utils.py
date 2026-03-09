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
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Optional, Any


# TODO use pooled connections to handle moderate to large traffic
@dataclass(frozen=True)
class SQLiteConnected:
    """A class that brings a connection to an SQLite database as a feature.

    Attributes
    ----------
    db_path : Path
        The file path to the SQLite database.
    """
    db_path: Optional[Path]
    timeout: float = 30.0

    def __setup_connection(self):
        conn = sqlite3.connect(
            self.db_path if self.db_path is not None else ":memory:", # use memory if path is none
            timeout=self.timeout,     # longer timeout
            check_same_thread=False)  # allows multi-threading

        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA journal_mode=WAL')    # use WAL mode for better concurrency on disk-backed DBs
        conn.execute('PRAGMA synchronous=NORMAL')  # accepts some crash vulnerabilities for improved performances
        conn.execute('PRAGMA cache_size=-64000')   # use a 64Mb cache

        return conn

    @contextmanager
    def _get_connection(self):
        """Get a context manager that handles the SQLite database connection.

        This method establishes a connection to the SQLite database specified by `db_path`,
        sets the row factory to `sqlite3.Row` for dictionary-like row access, and ensures
        the connection is properly closed after use.

        Yields
        ------
        sqlite3.Connection
            The SQLite database connection object.

        Example
        -------
        Whithin an object subclassing this feature, you can write
        >>> with self._get_connection() as conn:  # doctest: +SKIP
        ...     cursor = conn.cursor()
        ...     cursor.execute("SELECT * FROM some_table")
        """
        conn = self.__setup_connection()

        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def join(l: list[Any], sep: Any) -> list[Any]:
    """Interleave a separator between elements of a list.

    Parameters
    ----------
    l : list
        Sequence of elements to be joined. If empty, an empty list is returned.
    sep : Any
        Separator value to insert between consecutive elements of ``l``.

    Returns
    -------
    list
        A new list with ``sep`` inserted between each pair of elements from ``l``.
        If ``l`` is empty, returns ``[]``. If ``l`` contains a single element,
        a shallow copy of ``l`` is returned.

    Examples
    --------
    >>> join([1, 2, 3], 0)
    [1, 0, 2, 0, 3]
    >>> join(['a'], '-')
    ['a']
    >>> join([], None)
    []
    """

    if l:
        return list(chain.from_iterable((elem, sep) for elem in l[:-1])) + l[-1:]
    else:
        return []
