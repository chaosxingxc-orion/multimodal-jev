"""Compare actor-ready batches from two accepted behaviors of one policy."""
from .span_parity import compare_span_policies


def compare_batches(reference,resident,reference_selection,resident_selection):
    import torch
    if not reference or len(reference)!=len(resident) or reference_selection!=resident_selection:
        raise ValueError('same complete source selection required')
    worst=0.;ids=[]
    for old,new in zip(reference,resident,strict=True):
        for key in ('request_id','policy_id','candidates','request_weight'):
            if old[key]!=new[key]:raise ValueError('actor request contract differs: '+key)
        for key in ('positions','targets','supervised','span_weights'):
            if old[key].dtype!=new[key].dtype or not torch.equal(old[key],new[key]):
                raise ValueError('actor tensor differs: '+key)
        if old['inputs'].keys()!=new['inputs'].keys():raise ValueError('actor inputs differ')
        for key,left in old['inputs'].items():
            right=new['inputs'][key]
            if left.dtype!=right.dtype or not torch.equal(left,right):raise ValueError('encoded audio/input differs: '+key)
        audit=compare_span_policies([p[0].tolist() for p in old['behavior']],
                                   [p[0].tolist() for p in new['behavior']])
        if not audit['passed']:raise ValueError('same-policy behavior drift exceeds gate')
        worst=max(worst,audit['maximum_logprob_error']);ids.append(new['request_id'])
    if len(set(ids))!=len(ids):raise ValueError('distinct batch requests required')
    return dict(passed=True,requests=len(ids),request_ids=ids,maximum_logprob_error=worst,
        exact_inputs_and_supervision=True,scope='Actor input consumption only; no model forward, optimizer update or resident engine proof.')
