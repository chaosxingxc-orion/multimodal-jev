"""Check Linux pathname IPC length before allocating the inference engine."""
import os
from pathlib import Path


def check_ipc_directory(directory, *, suffix_bytes=36):
    # vLLM appends a UUID; leave the terminating NUL out of the 107-byte budget.
    raw=os.fspath(directory)
    if not Path(raw).is_absolute() or type(suffix_bytes) is not int or suffix_bytes<36:
        raise ValueError('absolute IPC directory and UUID-sized suffix required')
    count=len(os.fsencode(raw.rstrip('/')))+1+suffix_bytes
    if count>107:
        raise ValueError(f'IPC path needs {count} bytes, Linux limit is 107; bind local cache at a short container path')
    return dict(directory=raw,maximum_socket_path_bytes=count,limit_bytes=107)


def require_local_ipc():
    from .local_runtime_storage import check_paths
    raw=os.environ.get('VLLM_RPC_BASE_PATH') or os.environ.get('TMPDIR')
    if not raw:raise ValueError('explicit local IPC directory required')
    length=check_ipc_directory(raw)
    return dict(**length,storage=check_paths({'ipc':raw}))
