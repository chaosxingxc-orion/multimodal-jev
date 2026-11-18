"""Authenticate a dataset release once per content key, then reuse it.

`slurp_official_training.load` re-reads and re-hashes the whole release on every
call, and 60 modules call it, so the same 40,513 rows are authenticated dozens of
times per audit. This handle does not weaken that check: the key is built from a
freshly read manifest digest, the role/purpose pair and the implementation
digests of every module that participates in authentication. A different
manifest byte, a different reader version or a different role misses the key and
forces a full re-authentication.

Reused evidence is deep-copied on the way out, so a consumer that mutates a row
cannot poison the cache. Reuse is only ever for the same complete content, never
for a prior PASS or a timestamp.
"""
from pathlib import Path

from .baseline import file_hash
from .context_execution import content_key, singleflight

READERS = ('multimodal_jev/slurp_portable_release.py',
           'multimodal_jev/slurp_official_training.py',
           'multimodal_jev/slurp_training.py',
           'multimodal_jev/slurp_intent_structure.py',
           'multimodal_jev/task_schema.py')


def reader_digests():
    """Hash every module whose behaviour determines the authenticated result."""
    root = Path(__file__).resolve().parent
    digests = {}
    for name in READERS:
        path = root/Path(name).name
        if path.is_symlink() or not path.is_file():
            raise ValueError('release reader module missing: ' + name)
        digests[name] = file_hash(path)
    return digests


class ReleaseHandle:
    """One process-wide release cache keyed by complete content identity."""

    def __init__(self, *, max_entries=4):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError('positive bounded cache capacity required')
        self.flight = singleflight(max_entries=max_entries, max_running=1)
        self.readers = reader_digests()

    def key(self, root, manifest_sha256, role, purpose):
        root = Path(root)
        manifest = root/'MANIFEST.json'
        if manifest.is_symlink() or not manifest.is_file():
            raise ValueError('ordinary release manifest required')
        # Freshly read on every call; a cached digest is never trusted.
        if file_hash(manifest) != manifest_sha256:
            raise ValueError('release manifest changed before authentication')
        return content_key(dict(manifest_sha256=manifest_sha256, role=role,
                                purpose=purpose, root=str(root.resolve()),
                                readers=self.readers))

    def load(self, root, manifest_sha256, role, purpose):
        """Return (data, task, manifest, reused); evidence is a private copy.

        `AuditSingleFlight` already deep-copies on the way out, so each caller
        owns its rows and cannot poison the cached evidence.
        """
        from .slurp_portable_release import load
        key = self.key(root, manifest_sha256, role, purpose)
        value, reused = self.flight.get(
            key, lambda: load(Path(root), manifest_sha256, role, purpose=purpose))
        data, task, manifest = value
        return data, task, manifest, reused

    def stats(self):
        return dict(executions=self.flight.executions, hits=self.flight.hits,
                    entries=len(self.flight.entries), readers=self.readers)