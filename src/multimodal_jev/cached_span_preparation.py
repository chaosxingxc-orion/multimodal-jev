"""CPU-only span preparation for an isolated prefetch training candidate.

Workers prepare immutable input, not labels or policy. The consumer then binds
the current step policy and its unchanged scheduled supervision. Processor
instances are thread-local; the caller must authenticate processor assets.
"""
from dataclasses import dataclass
import threading

from .audio_policy import load_audio
from .baseline import file_hash
from .encoded_audio_cache import binding, canonical
from .span_contract import encode_span_request, span_labels


@dataclass(frozen=True)
class PreparedSpanInput:
    request: dict
    inputs: dict
    positions: object
    candidates: list
    audio_receipt: dict
    cache_hit: bool


class CachedSpanPreparer:
    def __init__(self, processor_factory, cache, *, processor_sha256, preprocessing=None):
        self.processor_factory = processor_factory
        self.cache = cache
        self.processor_sha256 = processor_sha256
        self.preprocessing = preprocessing
        self.local = threading.local()

    def __call__(self, job):
        request, expected_audio_sha256 = job
        # Retain source validation on hits initially; removing repeated reads
        # requires a separately authenticated immutable local-audio release.
        if file_hash(request.audio) != expected_audio_sha256:
            raise ValueError('source audio changed')
        identity = binding(request=request, audio_sha256=expected_audio_sha256,
                           processor_sha256=self.processor_sha256, preprocessing=self.preprocessing)
        value = self.cache.read(identity)
        hit = value is not None
        if not hasattr(self.local, 'processor'):
            self.local.processor = self.processor_factory()
        processor = self.local.processor
        if value is None:
            audio, receipt = load_audio(request.audio, self.preprocessing, return_receipt=True)
            inputs, positions = encode_span_request(processor, request, audio)
            self.cache.write(identity, inputs, positions, receipt)
        else:
            inputs, positions, receipt = value
        candidates = [m['token_ids'] for m in span_labels(processor.tokenizer, request.questions)]
        return PreparedSpanInput(canonical(request.model_input()), inputs, positions, candidates, receipt, hit)


def bind_instruction_slot(prepared, request, target, slot, *, epoch, batch_index, index, policy_id):
    """Bind unchanged supervision/weights after prefetch, at consumption time."""
    import torch
    if prepared.request != canonical(request.model_input()):
        raise ValueError('prepared request differs from current scheduled request')
    if not isinstance(policy_id, str) or not policy_id:
        raise ValueError('current policy identity required')
    if target['id'] != slot['id']:
        raise ValueError('target differs from scheduled recording')
    if (type(slot['padding']) is not bool or
            (slot['padding'] and slot['weight'] != 0)):
        raise ValueError('explicit zero-weight padding required')
    uid = f'{slot["id"]}::epoch{epoch}-batch{batch_index}-slot{index}'
    return dict(request_id=uid, policy_id=policy_id, inputs=prepared.inputs,
        positions=prepared.positions, candidates=prepared.candidates,
        targets=torch.tensor([target['targets']], dtype=torch.long),
        supervised=torch.tensor([target['supervised']], dtype=torch.bool),
        span_weights=torch.tensor([target['span_weights']], dtype=torch.float32),
        request_weight=slot['weight'], real_request=not slot['padding'])
