"""Scoped no-grad observation of native actor normalization, without replacement.

For isolated diagnostic processes only: the actor's imported function binding is
temporarily instrumented. CPU reference transfers/synchronization add overhead;
these timings are not uninstrumented inference throughput.
"""
from contextlib import contextmanager
import inspect
import itertools

from .baseline import file_hash

PINNED_SOURCE = '8ac62195a800ba784d1cddf03ee0bc7627f20236cc11e135fdba20e8d4a01f14'
PINNED_ACTOR_SOURCE = '207ba1d50a0344181779256b9bb32c733032c136e405d1846a57f9c78cda4cce'


def reference_logprobs(logits, labels, *, rows_per_chunk=16):
    """FP32 CPU reference from the exact incoming logits, with bounded copying.

    Copy by batch/sequence slices to avoid materializing a contiguous full
    response-vocabulary tensor on the accelerator. Retain all padded positions;
    the full-response exporter applies the authenticated response mask later.
    """
    import torch

    if (type(rows_per_chunk) is not int or rows_per_chunk < 1
            or logits.ndim not in (2, 3) or logits.shape[:-1] != labels.shape
            or logits.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or labels.dtype != torch.long or logits.device != labels.device
            or logits.numel() == 0):
        raise ValueError('nonempty aligned FP16/BF16/FP32 logits and integer labels required')
    target = labels.detach().to('cpu', copy=True)
    if not torch.all((target >= 0) & (target < logits.shape[-1])):
        raise ValueError('labels outside vocabulary')
    result = torch.empty(labels.shape, dtype=torch.float32)
    with torch.no_grad(), torch.autocast(device_type='cpu', enabled=False):
        for prefix in itertools.product(*(range(n) for n in logits.shape[:-2])):
            for start in range(0, logits.shape[-2], rows_per_chunk):
                index = prefix + (slice(start, start + rows_per_chunk),)
                chunk = logits[index].detach().to(device='cpu', dtype=torch.float32, copy=True)
                if not torch.isfinite(chunk).all():
                    raise ValueError('nonfinite input logits')
                result[index] = chunk.log_softmax(-1).gather(-1, target[index].unsqueeze(-1)).squeeze(-1)
    if not torch.isfinite(result).all():
        raise ValueError('nonfinite FP32 reference probabilities')
    return result


def observed_call(original, logits, labels, records, *, expected_device, rows_per_chunk=16,
                  **kwargs):
    """Call original exactly once, returning its exact object even if it differs.

    No tolerance, clipping or probability repair. Invalid numeric outputs raise;
    valid discrepancies are recorded for separately preregistered evaluation.
    """
    import torch

    if torch.is_grad_enabled():
        raise ValueError('normalization observer is diagnostic no-grad only')
    if logits.device.type != expected_device or labels.device.type != expected_device:
        raise ValueError('explicit expected probability device required')
    # Capture before native dispatch in case a future kernel mutates its inputs.
    reference = reference_logprobs(logits, labels, rows_per_chunk=rows_per_chunk)
    output = original(logits, labels, **kwargs)
    if (not isinstance(output, torch.Tensor) or output.shape != labels.shape
            or not output.is_floating_point() or output.device != logits.device):
        raise ValueError('native output must preserve probability shape and device')
    actual = output.detach().to(device='cpu', dtype=torch.float64, copy=True)
    if not torch.isfinite(actual).all() or (actual > 0).any():
        raise ValueError('native output has invalid log probabilities')
    delta = (actual - reference.double()).abs()
    records.append(dict(call=len(records), logits_shape=list(logits.shape),
        logits_dtype=str(logits.dtype), logits_device=str(logits.device),
        labels_dtype=str(labels.dtype), output_dtype=str(output.dtype), output_device=str(output.device),
        reference_dtype='torch.float32', reference_device='cpu',
        rows_per_chunk=rows_per_chunk, positions=labels.numel(),
        label_ids=labels.detach().cpu().tolist(),
        max_abs_error=delta.max().item(), mean_abs_error=delta.mean().item(),
        native_log_probs=actual.tolist(), reference_log_probs=reference.tolist(),
        scope='All response positions including padding; descriptive, no numerical acceptance threshold.'))
    return output


def bind_observations(data, log_probs, records, report, *, name, micro_batch_size):
    """Bind static, padded normalization calls to the authenticated input rows.

    Requires one call per contiguous microbatch and exact native-output equality
    on valid tokens. Padding is retained in call evidence but excluded from both
    exported runs. This is an evidence consistency check, not runtime attestation:
    identical token/probability rows alone cannot authenticate audio identity.
    The caller must collect the report around this exact isolated actor forward.
    """
    import torch
    from .omni_probability_capture import _aligned, capture
    from .sfc_reward_batch import digest

    rows = _aligned(data, records)
    if type(micro_batch_size) is not int or micro_batch_size < 1:
        raise ValueError('positive static microbatch size required')
    native_run = capture(data, log_probs, records, name=name)
    if (report.get('source_sha256') != PINNED_SOURCE
            or report.get('actor_source_sha256') != PINNED_ACTOR_SOURCE
            or report.get('flash_available') is not False
            or report.get('npu_available') is not True):
        raise ValueError('pinned native Ascend observer report required')
    calls = report.get('calls')
    count, width = data.batch['responses'].shape
    if type(calls) is not list or len(calls) != (count + micro_batch_size - 1) // micro_batch_size:
        raise ValueError('exactly one normalization call per static microbatch required')
    references = torch.empty((count, width), dtype=torch.float32)
    bindings = []
    for index, call in enumerate(calls):
        start = index * micro_batch_size; end = min(start + micro_batch_size, count)
        shape = call.get('logits_shape')
        if (type(call.get('call')) is not int or call['call'] != index
                or type(shape) is not list or len(shape) != 3
                or shape[:2] != [end - start, width]
                or type(shape[2]) is not int or shape[2] < 1
                or call.get('positions') != (end - start) * width
                or call.get('labels_dtype') != 'torch.int64'
                or call.get('reference_dtype') != 'torch.float32'
                or call.get('reference_device') != 'cpu'):
            raise ValueError('ordered padded normalization call shape required')
        device = call.get('logits_device', '')
        if (not (device == 'npu' or device.startswith('npu:'))
                or call.get('output_device') != device):
            raise ValueError('native NPU call devices required')
        labels = data.batch['responses'][start:end].detach().cpu()
        label_ids = call.get('label_ids')
        if (type(label_ids) is not list or len(label_ids) != end-start
                or any(type(row) is not list or len(row) != width
                       or any(type(token) is not int for token in row) for row in label_ids)
                or label_ids != labels.tolist()):
            raise ValueError('observed labels differ from exact padded response tokens')
        native = torch.tensor(call['native_log_probs'], dtype=torch.float64)
        reference = torch.tensor(call['reference_log_probs'], dtype=torch.float32)
        if (native.shape != labels.shape or reference.shape != labels.shape
                or not torch.isfinite(native).all() or not torch.isfinite(reference).all()
                or (native > 0).any() or (reference > 0).any()):
            raise ValueError('finite aligned observed probabilities required')
        mask = data.batch['response_mask'][start:end].detach().cpu().bool()
        returned = log_probs[start:end].detach().to(device='cpu', dtype=torch.float64)
        if not torch.equal(native[mask], returned[mask]):
            raise ValueError('native observations differ from returned actor probabilities')
        references[start:end] = reference
        bindings.append(dict(call=index,
            record_identities=[[r['example_id'], r['decision_id'], r['replicate']] for r in rows[start:end]],
            record_sha256=[digest(r) for r in rows[start:end]],
            response_mask=mask.tolist()))
    return dict(schema='omni-native-logprob-observation-v1',
        native=native_run,
        reference=capture(data, references, records, name=name+'-same-logits-fp32'),
        observer=report, microbatch_bindings=bindings,
        scope='Same-forward static microbatch consistency; no numerical acceptance or runtime attestation.')


@contextmanager
def observe_native_logprobs(*, expected_source_sha256=PINNED_SOURCE, rows_per_chunk=16):
    """Instrument the pinned, unmodified imported dispatcher on NPU only.

    Refuse CPU substitutes or nested observers. The original binding is restored
    on all exits; packages on disk are unchanged. Caller must exclude concurrent
    actor threads and bind call records to the actual microbatch/response order.
    """
    from verl.workers.actor import dp_actor
    from verl.utils import torch_functional

    original = dp_actor.logprobs_from_logits
    if file_hash(dp_actor.__file__) != PINNED_ACTOR_SOURCE:
        raise ValueError('pinned static microbatch actor source required')
    if (original is not torch_functional.logprobs_from_logits
            or file_hash(inspect.getfile(original)) != expected_source_sha256):
        raise ValueError('unchanged pinned native probability dispatcher required')
    if type(rows_per_chunk) is not int or rows_per_chunk < 1:
        raise ValueError('positive reference chunk size required')
    flash = torch_functional.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE
    npu = torch_functional.NPU_CROSS_ENTROPY_LOSS_AVAILABLE
    if flash or not npu:
        raise ValueError('native Ascend cross entropy dispatch required')
    report = dict(source_sha256=expected_source_sha256, actor_source_sha256=PINNED_ACTOR_SOURCE,
        dispatcher=original.__module__+'.'+original.__qualname__,
        flash_available=bool(flash), npu_available=bool(npu), calls=[])

    def instrumented(logits, labels, inplace_backward=True):
        return observed_call(original, logits, labels, report['calls'], expected_device='npu',
                             rows_per_chunk=rows_per_chunk, inplace_backward=inplace_backward)

    dp_actor.logprobs_from_logits = instrumented
    try:
        yield report
    finally:
        dp_actor.logprobs_from_logits = original
