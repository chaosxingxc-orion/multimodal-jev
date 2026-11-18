"""Read an authenticated preprocessing cache without creating or filling entries."""
from pathlib import Path

from .encoded_audio_cache import EncodedAudioCache


class ReadOnlyEncodedAudioCache(EncodedAudioCache):
    def __init__(self, root):
        self.root = Path(root)
        if not self.root.is_dir():
            raise ValueError('prebuilt input cache required')

    def read(self, identity):
        value = super().read(identity)
        if value is None:
            raise ValueError('prebuilt input cache miss; rebuild offline')
        return value

    def write(self, *args, **kwargs):
        raise ValueError('prebuilt input cache is read-only')
