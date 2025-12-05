from dataclasses import dataclass
from typing import Any, Optional
from pathlib import Path


class NotConfiguredError(Exception):
    def __init__(self, name: str):
        super().__init__(f"The element {name} is not configured!")


@dataclass
class __Config:
    db_path: Optional[Path] = None
    data_path: Optional[Path] = None
    upload_path: Optional[Path] = None
    max_processing_queue_workers: int = 1

    def __getattribute__(self, name: str) -> Any:
        if value := super().__getattribute__(name):
            return value
        else:
            raise NotConfiguredError(name)

    def __repr__(self) -> str:
        fields = []
        for field in self.__dataclass_fields__:
            try:
                value = object.__getattribute__(self, field)
                fields.append(f"{field}={value}")
            except AttributeError:
                fields.append(f"{field}=None")
        return f"Config({', '.join(fields)})"



config = __Config()
