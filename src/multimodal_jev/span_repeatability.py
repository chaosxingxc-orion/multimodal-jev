"""Same-engine trace intervention diagnostic; no policy admission."""
import torch

REPEAT_SEQUENCE = ('plain-before-1', 'plain-before-2', 'trace', 'plain-after-1', 'plain-after-2')


def compare_repeats(records):
    if [r['phase'] for r in records] != list(REPEAT_SEQUENCE):
        raise ValueError('fixed before/trace/after sequence required')
    tensors = [torch.tensor(r['candidate_logits'], dtype=torch.float32) for r in records]
    if any(t.shape != tensors[0].shape or not torch.isfinite(t).all() for t in tensors):
        raise ValueError('finite aligned repeated logits required')
    pairs = []
    for i in range(len(records)):
        for j in range(i+1, len(records)):
            a, b = tensors[i], tensors[j]
            pairs.append(dict(left=records[i]['phase'], right=records[j]['phase'],
                logits_exact=bool(torch.equal(a,b)), maximum_logit_difference=float((a-b).abs().max()),
                maximum_logprob_difference=float((a.log_softmax(-1)-b.log_softmax(-1)).abs().max())))
    return dict(all_logits_exact=all(p['logits_exact'] for p in pairs), pairs=pairs,
                scope='One request in one engine; does not prove cross-engine determinism or HF parity')


def run_repeat_probe(engine, encode, output):
    from .baseline import atomic_json, file_hash
    records=[]
    for phase in REPEAT_SEQUENCE:
        if phase=='trace':
            engine.llm_engine.collective_rpc('begin_jev_span_trace')
        logits=encode()
        if phase=='trace':
            receipts=engine.llm_engine.collective_rpc('finish_jev_span_trace_ranked',
                args=(str(output/'repeat-trace.pt'),))
            if sorted(r['rank'] for r in receipts)!=list(range(4)):
                raise ValueError('four trace rank receipts required')
            atomic_json(output/'repeat-trace-files.json',dict(workers=receipts,
                files={r['path']:file_hash(r['path']) for r in receipts}))
        records.append(dict(phase=phase,candidate_logits=logits.tolist()))
    return dict(records=records,comparison=compare_repeats(records))
