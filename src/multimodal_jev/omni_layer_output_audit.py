"""Descriptive first-layer tensor comparisons; never a policy parity gate."""
import math


def validate_output(saved, original):
    import torch
    if set(saved) != {'output', 'token_mask'}:
        raise ValueError('complete output and original mask required')
    output = saved['output']; mask = saved['token_mask']
    hidden = original['hidden_states']; expected_mask = original['token_mask']
    if (output.shape != hidden.shape or output.dtype != torch.bfloat16
            or hidden.dtype != torch.bfloat16 or not torch.isfinite(output).all()
            or mask.shape != hidden.shape[:2] or mask.dtype != expected_mask.dtype
            or not torch.equal(mask, expected_mask)
            or not torch.all((mask == 0) | (mask == 1))
            or not torch.all(mask.sum(-1) > 0)):
        raise ValueError('output shape/dtype/finiteness or original token mask differs')


def difference(left, right):
    import torch
    if (left.shape != right.shape or left.dtype != right.dtype or left.numel() == 0
            or not torch.isfinite(left).all() or not torch.isfinite(right).all()):
        raise ValueError('same nonempty finite comparison tensors required')
    delta = (left.double() - right.double()).abs()
    return dict(equal=bool(torch.equal(left, right)), elements=left.numel(),
        differing_elements=int((left != right).sum()), max_abs_error=float(delta.max()),
        mean_abs_error=float(delta.mean()), rms_error=float(delta.square().mean().sqrt()))


def paired_rows(single, batched, index):
    """Keep padded positions visible; valid-token statistics cannot hide them."""
    import torch
    mask = single['token_mask'][0]
    if not torch.equal(mask, batched['token_mask'][index]):
        raise ValueError('paired original masks differ')
    left = single['output'][0]; right = batched['output'][index]
    return dict(valid_tokens=int(mask.sum()),
        valid=difference(left[mask.bool()], right[mask.bool()]),
        full_padded=difference(left, right))


def validate_accounting(records):
    if [(r['size'], r['offset']) for r in records] != [(s,o) for s in (1,2) for o in range(0,6,s)]:
        raise ValueError('all nine ordered calls required')
    previous_allocated = previous_reserved = 0
    for record in records:
        allocated = record['peak_allocated_bytes']; reserved = record['peak_reserved_bytes']
        if (type(allocated) is not int or type(reserved) is not int
                or not 0 < allocated <= reserved or allocated < previous_allocated
                or reserved < previous_reserved or type(record['seconds']) not in (int,float)
                or not math.isfinite(record['seconds']) or record['seconds'] < 0):
            raise ValueError('finite duration and cumulative allocator peaks required')
        previous_allocated, previous_reserved = allocated, reserved
    return dict(peak_allocated_bytes=previous_allocated, peak_reserved_bytes=previous_reserved,
        forward_and_save_seconds=sum(r['seconds'] for r in records),
        scope='PyTorch cumulative allocator peaks; not total device HBM or full-model capacity.')
