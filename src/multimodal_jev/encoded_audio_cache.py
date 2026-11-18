"""Policy-independent CPU input cache for deterministic audio preprocessing.

The caller authenticates source/processor files at launch, supplies their content
hashes and binds the current policy and labels AFTER lookup. No model states,
targets, sampling results, or policy IDs are accepted in cache payloads.
Use local /data in deployment. This module is not wired into the live trainer.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch

from .span_request_identity import tensor_identity


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False))


def binding(*, request, audio_sha256, processor_sha256, preprocessing):
    for digest in (audio_sha256, processor_sha256):
        if (type(digest) is not str or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('authenticated audio and processor SHA256 required')
    return canonical(dict(schema='encoded-audio-cache-v1', request=request.model_input(),
                          audio_sha256=audio_sha256, processor_sha256=processor_sha256,
                          preprocessing=preprocessing))


def key_for(identity):
    return hashlib.sha256(json.dumps(canonical(identity), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class EncodedAudioCache:
    """Immutable entries, atomic publication, corruption fails closed.

    Each read returns independent CPU tensors: batch collation may mutate its own
    copy without corrupting future epochs. Optional metadata never enters inputs.
    Concurrent writers must produce identical entries for the same binding.
    """
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate(inputs, positions):
        required = {'input_ids', 'attention_mask', 'input_features', 'input_features_mask'}
        if not required <= inputs.keys() or inputs.keys() - required - {'position_ids'}:
            raise ValueError('only label-free processor tensors allowed')
        for tensor in [*inputs.values(), positions]:
            if (not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu'
                    or tensor.requires_grad or not torch.isfinite(tensor).all()):
                raise ValueError('finite detached CPU tensors required')
        ids, mask = inputs['input_ids'], inputs['attention_mask']
        if (ids.dtype != torch.long or ids.ndim != 2 or ids.shape[0] != 1 or mask.shape != ids.shape
                or not ((mask == 0) | (mask == 1)).all()):
            raise ValueError('one complete input and binary mask required')
        if (positions.dtype != torch.long or positions.ndim != 2 or positions.shape[0] != 1
                or positions.shape[1] == 0 or ((positions < 0) | (positions >= ids.shape[1])).any()
                or not mask.gather(1, positions).all()
                or (positions[:, 1:] <= positions[:, :-1]).any()):
            raise ValueError('ordered valid readout positions required')

    def read(self, identity):
        path = self.root / (key_for(identity)+'.pt')
        if not path.exists():
            return None
        value = torch.load(path, map_location='cpu', weights_only=True)
        if set(value) != {'binding', 'inputs', 'positions', 'audio_receipt', 'tensor_hashes'}:
            raise ValueError('unexpected cache payload')
        if value['binding'] != canonical(identity):
            raise ValueError('cache binding differs')
        self._validate(value['inputs'], value['positions'])
        actual = {k:tensor_identity(v) for k,v in dict(value['inputs'], positions=value['positions']).items()}
        if actual != value['tensor_hashes']:
            raise ValueError('cache tensor integrity differs')
        return value['inputs'], value['positions'], value['audio_receipt']

    def write(self, identity, inputs, positions, audio_receipt):
        self._validate(inputs, positions)
        tensors = {k:v.detach().clone().contiguous() for k,v in inputs.items()}
        positions = positions.clone().contiguous()
        value = dict(binding=canonical(identity), inputs=tensors, positions=positions,
                     audio_receipt=canonical(audio_receipt),
                     tensor_hashes={k:tensor_identity(v) for k,v in dict(tensors, positions=positions).items()})
        path = self.root / (key_for(identity)+'.pt')
        fd, temp = tempfile.mkstemp(prefix='.building-', dir=self.root)
        try:
            with os.fdopen(fd, 'wb') as stream:
                torch.save(value, stream)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temp, path)  # atomic, no replacement of another writer
                directory_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
                try: os.fsync(directory_fd)
                finally: os.close(directory_fd)
            except FileExistsError:
                existing = self.read(identity)
                if (existing[2] != value['audio_receipt'] or
                        any(not torch.equal(existing[0][k], v) for k,v in tensors.items()) or
                        not torch.equal(existing[1], positions)):
                    raise ValueError('non-deterministic cache collision')
        finally:
            Path(temp).unlink(missing_ok=True)
        return path
