"""Durable commits on admitted local storage; never disable fsync globally."""
import json
import os
from pathlib import Path
import tempfile

from .local_runtime_storage import check_paths


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def commit_file(path, writer):
    """Publish only a flushed, synced file; writer receives a binary stream."""
    path = Path(path)
    check_paths({'checkpoint': path})
    if path.exists():
        raise FileExistsError('immutable checkpoint already exists: ' + str(path))
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            writer(stream)
            os.fchmod(stream.fileno(), 0o644)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def commit_json(path, value):
    payload = (json.dumps(value, indent=2) + '\n').encode()
    commit_file(path, lambda stream: stream.write(payload))


def seal_adapter(path):
    """Sync a newly exported adapter before its identity is published."""
    path = Path(path)
    check_paths({'adapter': path})
    for name in ('adapter_model.safetensors', 'adapter_config.json'):
        with (path / name).open('rb') as stream:
            # safetensors may create mode0600 even under a readable umask.
            # Publish model artifacts readable by host-side independent audits.
            os.fchmod(stream.fileno(), 0o644)
            os.fsync(stream.fileno())
    sync_directory(path)
    sync_directory(path.parent)
