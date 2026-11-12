"""Reconstruct saved correctness supervision without importing torch.

This checks the saved training payload, not the gradient actually consumed by
the optimizer. Runtime/source and tensor-transport evidence remain separate.
"""
import json
import math
import struct


def float32(value):
    if type(value) not in (int,float) or not math.isfinite(value):
        raise ValueError('finite numeric Span weight required')
    return struct.unpack('f',struct.pack('f',value))[0]


def verify(selected, behavior, saved, arm):
    if arm not in ('D','Q','P') or not selected:
        raise ValueError('nonempty admitted D/Q/P selection required')
    rows={r['id']:r for r in behavior['records']}
    if len(rows)!=len(behavior['records']) or len({r['id'] for r in selected})!=len(selected):
        raise ValueError('distinct behavior and selected identities required')
    wire='D' if arm=='D' else 'Q'
    expected=[];weights=[1.] if arm=='D' else ([.1,.1,1.] if arm=='P' else [0.,0.,1.])
    for item in selected:
        target=item['target'];row=rows[item['id']+'::'+wire]
        if (target['span_weights']!=weights or target['supervised']!=[True]*len(weights)
                or any(type(x) is not bool for x in target['supervised'])
                or len(target['targets'])!=len(weights)
                or len(row['candidate_logprobs'])!=len(weights)
                or any(type(y) is not int or not 0<=y<len(p)
                       for y,p in zip(target['targets'],row['candidate_logprobs'],strict=True))):
            raise ValueError('registered correctness targets or Span budget differ')
        rw=target['request_weight']
        if type(rw) not in (int,float) or not math.isfinite(rw) or rw<=0:
            raise ValueError('positive finite source weight required')
        expected.append(dict(request_id=row['request_id'],targets=[target['targets']],
            supervised=[target['supervised']],span_weights=[[float32(x) for x in weights]],
            request_weight=rw))
    if len({r['request_id'] for r in expected})!=len(expected):
        raise ValueError('distinct bound requests required')
    # JSON comparison distinguishes bool/int and keeps exact stored float values.
    encode=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
    if encode(saved)!=encode(expected):
        raise ValueError('saved supervision differs from independently admitted targets')
    return dict(passed=True,arm=arm,requests=len(expected),span_weights=weights,
        stored_float32_span_weights=[float32(x) for x in weights],
        scope='Saved correctness supervision matches admitted labels and accepted behavior request IDs. '
              'Not independent proof of executed advantages, gradients or process-reward efficacy.')
