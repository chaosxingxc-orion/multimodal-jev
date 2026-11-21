"""Content-addressed local audio replicas, preserving logical request paths.

After publication, the pinned content hash identifies the admitted dataset
version. Resolving a replica never reopens the shared source or silently falls
back to it. The original request and encoded-input cache identity stay unchanged.
"""
import fcntl
import hashlib
from pathlib import Path

from .baseline import file_hash
from .local_checkpoint import commit_file
from .local_runtime_storage import check_paths,mount_for,LOCAL_FILESYSTEMS


class ReadOnlyAudioMirror:
    """Read published content by hash, including Ceph behind a read-only bind."""
    def __init__(self, root):
        self.root = Path(root)
        mount = mount_for(self.root, Path('/proc/self/mountinfo').read_text())
        if (self.root.is_symlink() or not self.root.is_dir()
                or not (mount['filesystem'] in LOCAL_FILESYSTEMS
                        or (mount['filesystem'] == 'ceph' and 'ro' in mount['options'].split(',')))):
            raise ValueError('published audio directory on local disk or read-only Ceph required')

    def _path(self, digest):
        if (type(digest) is not str or len(digest) != 64 or
                any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('explicit SHA256 audio identity required')
        return self.root / (digest + '.audio')

    def resolve(self, digest):
        path = self._path(digest)
        if path.is_symlink() or not path.is_file() or file_hash(path) != digest:
            raise ValueError('mirrored audio missing or changed: ' + digest)
        return path


class LocalAudioMirror(ReadOnlyAudioMirror):
    def __init__(self, root, *, read_only=False):
        self.root = Path(root)
        self.read_only = read_only
        if read_only:
            mount=mount_for(self.root,Path('/proc/self/mountinfo').read_text())
            if mount['filesystem'] not in LOCAL_FILESYSTEMS or not self.root.is_dir():
                raise ValueError('existing local audio mirror required')
        else:
            check_paths({'audio_mirror': self.root})
            self.root.mkdir(parents=True, exist_ok=True)

    def publish(self, source, digest):
        if self.read_only:
            raise ValueError('read-only audio mirror cannot publish')
        path = self._path(digest)
        # Serialize publication across processes; never overwrite an existing blob.
        with (self.root / '.publish.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists() or path.is_symlink():
                return self.resolve(digest)
            def copy_verified(output):
                actual = hashlib.sha256()
                with Path(source).open('rb') as stream:
                    while chunk := stream.read(1024 * 1024):
                        output.write(chunk)
                        actual.update(chunk)
                if actual.hexdigest() != digest:
                    raise ValueError('source audio differs from admitted identity')
            commit_file(path, copy_verified)
            return self.resolve(digest)
