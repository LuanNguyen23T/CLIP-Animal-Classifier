import os
import uuid
from typing import BinaryIO
from app.storage.base import Storage

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "../../../uploads")

class LocalStorage(Storage):

    def __init__(self, upload_dir: str = UPLOAD_DIR):
        self.upload_dir = upload_dir
        os.makedirs(self.upload_dir, exist_ok=True)

    def save(self, file: BinaryIO, filename: str) -> str:
        """Save uploaded file, return unique storage key."""
        ext = os.path.splitext(filename)[-1].lower()
        unique_key = f"{uuid.uuid4().hex}{ext}"
        dest_path = os.path.join(self.upload_dir, unique_key)
        with open(dest_path, "wb") as f:
            f.write(file.read())
        return unique_key

    def get_path(self, key: str) -> str:
        """Return absolute local path for a given key."""
        return os.path.join(self.upload_dir, key)

    def delete(self, key: str) -> None:
        """Delete file by key, silently ignore if not found."""
        path = self.get_path(key)
        if os.path.exists(path):
            os.remove(path)
