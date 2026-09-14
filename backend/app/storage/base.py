# storage/base.py
from abc import ABC, abstractmethod
from typing import BinaryIO

class Storage(ABC):

    @abstractmethod
    def save(self, file: BinaryIO, filename: str) -> str:
        """Save a file and return its storage key (relative filename)."""
        pass

    @abstractmethod
    def get_path(self, key: str) -> str:
        """Resolve a storage key to an absolute local path."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a file by its storage key."""
        pass
