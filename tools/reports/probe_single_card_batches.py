"""Label-free comparisons of frozen evaluation batches and independent HF calls."""
from itertools import islice
import time

import torch

from tools.reports.evaluate_single_cards import pair_geometry


def geometry(item):
    tokens, padded, valid = pair_geometry(item)
    return dict(tokens=tokens, padded_frames=padded, valid_frames=valid)


@torch.inference_mode()
def compare(scorer, model, device, provider, policy, rows, spec):
    records = []
    sliced_trace = False
    def forward(group):
        start = time.monotonic()
        out = scorer.forward(model, device, provider, policy, group)
        return out.float().flatten().cpu(), out.sigmoid().float().flatten().cpu(), time.monotonic()-start

    candidate = scorer.evaluation_forward(model, device, provider, policy) if scorer.mode != 'bounded_batch' else None

    for group in islice(scorer.groups(policy, rows, {}), spec['batches_per_block']):
        if not spec['edge_and_middle']:
            indices = [0]
        else:
            indices = sorted({0, len(group)//2, len(group)-1})
        batch_z, batch_p, batch_seconds = forward(group)
        if candidate is not None:
            start = time.monotonic()
            out = candidate(group)
            candidate_z, candidate_p = out.float().flatten().cpu(), out.sigmoid().float().flatten().cpu()
            candidate_seconds = time.monotonic()-start
        if spec.get('compare_reuse') and scorer.mode.startswith('reuse_audio_'):
            from multimodal_jev.card_acoustic_reuse import score
            # The same original inputs, including the cases where bounded batches disagree.
            alt_start = time.monotonic()
            alt = score(model, group, device, policy, scorer.collate, split_lengths=True)
            alt_z, alt_p = alt.float().flatten().cpu(), alt.sigmoid().float().flatten().cpu()
            alt_seconds = time.monotonic() - alt_start
        examples = []
        shapes = [geometry(item) for item in group]
        for i in indices:
            item = group[i]
            uid = item['request_id'].rsplit(':', 1)[0]
            z, p, seconds = forward([item])
            repeated_z, repeated_p, _ = forward([item])
            if not torch.equal(z, repeated_z) or not torch.equal(p, repeated_p):
                raise ValueError(f'independent HF is not repeatable for {item["request_id"]}')
            variants = {}
            for name, members in (
                ('same_audio', [j for j, x in enumerate(group) if x['request_id'].rsplit(':', 1)[0] == uid]),
                ('same_shape', [j for j, shape in enumerate(shapes) if shape == shapes[i]]),
            ):
                if len(members) in (1, len(group)):
                    variant_z, variant_p, elapsed = (z[0], p[0], seconds) if len(members) == 1 else (
                        batch_z[i], batch_p[i], batch_seconds)
                else:
                    logits, probs, elapsed = forward([group[j] for j in members])
                    k = members.index(i)
                    variant_z, variant_p = logits[k], probs[k]
                variants[name] = dict(pairs=len(members), logit=variant_z.item(),
                                      probability=variant_p.item(), seconds=elapsed)
            if candidate is not None:
                variants[scorer.mode] = dict(pairs=len(group), logit=candidate_z[i].item(),
                                            probability=candidate_p[i].item(), seconds=candidate_seconds)
            if spec.get('compare_reuse') and scorer.mode.startswith('reuse_audio_'):
                variants['reuse_audio_length'] = dict(pairs=len(group), logit=alt_z[i].item(),
                    probability=alt_p[i].item(), seconds=alt_seconds)
            examples.append(dict(index=i, id=item['request_id'], singleton_logit=z.item(),
                singleton_probability=p.item(), singleton_seconds=seconds, batch_logit=batch_z[i].item(),
                batch_probability=batch_p[i].item(), variants=variants))
        trace_data = {}
        if spec.get('trace_padding') and not sliced_trace and candidate is not None:
            for example in examples:
                if example['variants'][scorer.mode]['logit'] == example['singleton_logit']:
                    continue
                from tools.reports.trace_card_padding import trace
                trace_data['trace'] = trace(model, group, example['index'], candidate,
                    lambda one: scorer.forward(model, device, provider, policy, one))
                sliced_trace = True
                break
        records.append(dict(**trace_data, pairs=len(group), batch_seconds=batch_seconds, examples=examples,
                            neighbors=[dict(id=x['request_id'], **g) for x, g in zip(group, shapes, strict=True)]))
    if not records:
        raise ValueError('empty single-card comparison')
    return records


@torch.inference_mode()
def verify(scorer, model, device, provider, policy, rows, expected, diagnostic_mismatches=None):
    forward = scorer.evaluation_forward(model, device, provider, policy)
    groups = scorer.groups(policy, rows, {})
    for group, receipt in zip(islice(groups, len(expected)), expected, strict=True):
        neighbors = [dict(id=item['request_id'], **geometry(item)) for item in group]
        if neighbors != receipt['neighbors']:
            raise ValueError('HF panel identity or geometry changed')
        if scorer.mode == 'hf_singleton':
            # Only execute the actual reference entries during admission.
            values = {}
            for row in receipt['examples']:
                output = forward([group[row['index']]])
                values[row['index']] = output.item(), output.sigmoid().item()
        else:
            output = forward(group)
            prob = output.sigmoid()
            values = {row['index']: (output[row['index'], 0].item(), prob[row['index'], 0].item())
                      for row in receipt['examples']}
        for row in receipt['examples']:
            if group[row['index']]['request_id'] != row['id']:
                raise ValueError('pinned HF candidate identity changed')
            if values[row['index']] != (row['singleton_logit'], row['singleton_probability']):
                if diagnostic_mismatches is None or scorer.mode != 'bounded_batch':
                    raise ValueError(f'worker {scorer.mode} differs from pinned HF reference: {row["id"]}')
                diagnostic_mismatches.append(row['id'])
    return sum(len(receipt['examples']) for receipt in expected)
