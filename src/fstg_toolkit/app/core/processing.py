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

from typing import TypeVar, Callable, Optional, Dict
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
import sqlite3
from contextlib import contextmanager


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


T = TypeVar('T')


class ProcessingQueueListener(ABC):
    @abstractmethod
    def on_job_submitted(self, job_id: str):
        pass

    @abstractmethod
    def on_job_started(self, job_id: str):
        pass

    @abstractmethod
    def on_job_completed(self, job_id: str, result: Optional[T]):
        pass

    @abstractmethod
    def on_job_failed(self, job_id: str, error: Exception):
        pass


class ProcessingQueue:
    def __init__(self, max_workers: int = 1, listener: Optional[ProcessingQueueListener] = None):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='ProcessingQueueWorkers')
        self.futures: Dict[str, Future] = {}
        self.listener = listener

    def __process(self, job_id: str, func: Callable[..., T], *args, **kwargs) -> T:
        try:
            if self.listener:
                self.listener.on_job_started(job_id)

            result = func(*args, **kwargs)

            if self.listener:
                self.listener.on_job_completed(job_id, result)

            return result
        except Exception as ex:
            if self.listener:
                self.listener.on_job_failed(job_id, ex)
            raise ex

    def submit(self, func: Callable[..., T], *args, **kwargs) -> str:
        job_id = str(uuid.uuid4())
        self.futures[job_id] = self.executor.submit(self.__process, job_id, func, *args, **kwargs)

        if self.listener:
            self.listener.on_job_submitted(job_id)

        return job_id

    def get_result(self, job_id: str) -> Optional[T]:
        if future := self.futures.get(job_id):
            if future.done():
                return future.result()
        return None


class ProcessingJobStatus(Enum):
    PENDING = "Pending"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"


class JobStatusMonitor(ProcessingQueueListener):
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.__initialize_db()

    def __initialize_db(self):
        with self.__get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT UNIQUE,
                    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT,
                    result TEXT,
                    error TEXT,
                    PRIMARY KEY(id)
                )
            ''')

    @contextmanager
    def __get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def on_job_submitted(self, job_id: str):
        with self.__get_connection() as conn:
            conn.execute('''
                INSERT INTO jobs (id, status)
                VALUES (?, ?)
            ''', (job_id, ProcessingJobStatus.PENDING.name))

    def __update(self, job_id: str, **kwargs):
        with self.__get_connection() as conn:
            updates = [f'{field} = ?' for field in kwargs]
            values = tuple([kwargs[field] for field in kwargs] + [job_id])
            conn.execute(f'''
                UPDATE jobs SET {', '.join(updates)}
                WHERE id = ?
            ''', values)

    def on_job_started(self, job_id: str):
        self.__update(job_id, status=ProcessingJobStatus.RUNNING.name)

    def on_job_completed(self, job_id: str, result: Optional[T]):
        self.__update(job_id, status=ProcessingJobStatus.COMPLETED.name, result=str(result))

    def on_job_failed(self, job_id: str, error: Exception):
        self.__update(job_id, status=ProcessingJobStatus.FAILED.name, error=str(error))

    def list_jobs(self, limit: int = 30) -> list[dict[str, str]]:
        with self.__get_connection() as conn:
            rows = conn.execute(f'''
                SELECT * FROM jobs
                ORDER BY submitted_at DESC
                LIMIT ?
            ''', (limit,)).fetchall()
            return [dict(row) for row in rows]


singleton_processing_queue: Optional[ProcessingQueue] = None


def init_processing_queue(max_workers: int = 1, listener: Optional[ProcessingQueueListener] = None):
    global singleton_processing_queue
    singleton_processing_queue = ProcessingQueue(max_workers=max_workers, listener=listener)


def get_processing_queue() -> ProcessingQueue:
    global  singleton_processing_queue

    if not singleton_processing_queue:
        init_processing_queue()

    return singleton_processing_queue
