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

import shutil
import subprocess
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TypeVar, Callable, Optional, Dict, Any

from .config import config
from .datafilesdb import get_data_file_db
from .utils import SQLiteConnected

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

    def __process(self, job_id: str, func: Callable[[str, ...], T], *args, **kwargs) -> T:
        try:
            if self.listener:
                self.listener.on_job_started(job_id)

            result = func(job_id, *args, **kwargs)

            if self.listener:
                self.listener.on_job_completed(job_id, result)

            return result
        except Exception as ex:
            if self.listener:
                self.listener.on_job_failed(job_id, ex)
            raise ex

    def submit(self, func: Callable[[str, ...], T], *args, **kwargs) -> str:
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

    @staticmethod
    def from_value(name: str) -> 'Optional[ProcessingJobStatus]':
        for status in ProcessingJobStatus:
            if name == status.name:
                return status
        return None


class JobStatusMonitor(SQLiteConnected, ProcessingQueueListener):
    def __init__(self, db_path: Path):
        super().__init__(db_path)
        self.__init_db()

    def __init_db(self):
        with self._get_connection() as conn:
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

    def on_job_submitted(self, job_id: str):
        with self._get_connection() as conn:
            conn.execute('''
                INSERT INTO jobs (id, status)
                VALUES (?, ?)
            ''', (job_id, ProcessingJobStatus.PENDING.name))

    def __update(self, job_id: str, **kwargs):
        with self._get_connection() as conn:
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


singleton_processing_queue: Optional[ProcessingQueue] = None


def get_processing_queue() -> ProcessingQueue:
    global  singleton_processing_queue

    if not singleton_processing_queue and config.is_configured('db_path'):
        monitor = JobStatusMonitor(db_path=config.db_path)
        singleton_processing_queue = ProcessingQueue(
            max_workers=config.max_processing_queue_workers, listener=monitor)

    return singleton_processing_queue


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

    def check(self):
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

    @staticmethod
    def from_record(record: dict[str, Any]) -> 'SubmittedDataset':
        return SubmittedDataset(
            name=record['name'],
            include_raw=bool(record['include_raw']),
            compute_metrics=bool(record['compute_metrics']),
            areas_file=Path(record['areas_path']),
            matrices_files=[Path(s) for s in record['matrices_paths'].split(';')])


def _process_dataset(job_id: str, dataset: SubmittedDataset) -> Optional[str]:
    try:
        output_path = config.data_path / f'{job_id}.zip'

        # compute the model from all sequences of matrices
        command = ['python', '-m', 'fstg_toolkit', 'graph', 'build',
                   '--max-cpus', str(config.max_processing_cpus),
                   '-o', str(output_path)]

        if not dataset.include_raw:
            command.append('--no-raw')

        command += [str(dataset.areas_file),
                    *[str(file) for file in dataset.matrices_files]]
        subprocess.run(command, check=True, capture_output=True)

        # compute the metrics
        if dataset.compute_metrics:
            command = ['python', '-m', 'fstg_toolkit', 'graph', 'metrics',
                       '--max-cpus', str(config.max_processing_cpus),
                       str(output_path)]
            subprocess.run(command, check=True, capture_output=True)

        # register output to get a token
        return get_data_file_db().add(output_path)
    finally:
        # TODO we could use async functions for IO (at least files) to improve performances
        # clean input files
        shutil.rmtree(dataset.areas_file.parent)
        for p in {file.parent for file in dataset.matrices_files}:
            shutil.rmtree(p)


@dataclass(frozen=True)
class DatasetResult:
    dataset: SubmittedDataset
    job_status: ProcessingJobStatus
    submitted_at: datetime
    result: str
    error: Optional[str]


class DatasetProcessingManager(SQLiteConnected):
    def __init__(self, db_path: Path):
        super().__init__(db_path)

        # NOTE Triggers processing queue and monitoring creation if not
        # already created. It is necessary as the processing manager
        # depends on monitoring database.
        _ = get_processing_queue()

        self.__init_db()

    def __init_db(self):
        with self._get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER UNIQUE,
                    name TEXT NOT NULL,
                    include_raw INTEGER NOT NULL,
                    compute_metrics INTEGER NOT NULL,
                    areas_path TEXT NOT NULL,
                    matrices_paths TEXT NOT NULL,
                    job_id TEXT UNIQUE,
                    PRIMARY KEY(id)
                    FOREIGN KEY(job_id) REFERENCES jobs(id)
                )
            ''')

    def __insert_record(self, dataset: SubmittedDataset, job_id: str):
        with self._get_connection() as conn:
            conn.execute('''
                INSERT INTO datasets (name, include_raw, compute_metrics, areas_path, matrices_paths, job_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (dataset.name, int(dataset.include_raw), int(dataset.compute_metrics),
                  str(dataset.areas_file), ";".join(str(p) for p in dataset.matrices_files), job_id))

    def submit(self, dataset: SubmittedDataset):
        dataset.check()
        job_id = get_processing_queue().submit(_process_dataset, dataset)
        self.__insert_record(dataset, job_id)

    def list(self, limit: int = 30) -> list[DatasetResult]:
        # NOTE for performances and practical reasons, this class and the job monitoring
        # one are coupled into this join SQL request.
        with self._get_connection() as conn:
            rows = conn.execute('''
                SELECT * FROM datasets A
                INNER JOIN jobs B ON A.job_id = B.id
                ORDER BY B.submitted_at DESC
                LIMIT ?
            ''', (limit,)).fetchall()
            rows = [dict(row) for row in rows]
            return [DatasetResult(
                        dataset=SubmittedDataset.from_record(row),
                        job_status=ProcessingJobStatus.from_value(row['status']),
                        submitted_at=datetime.fromisoformat(row['submitted_at']).replace(tzinfo=timezone.utc),
                        result=row['result'],
                        error=row['error'])
                    for row in rows]


singleton_processing_manager: Optional[DatasetProcessingManager] = None


def get_dataset_processing_manager() -> DatasetProcessingManager:
    global singleton_processing_manager

    if not singleton_processing_manager and config.is_configured('db_path'):
        singleton_processing_manager = DatasetProcessingManager(config.db_path)

    return singleton_processing_manager
