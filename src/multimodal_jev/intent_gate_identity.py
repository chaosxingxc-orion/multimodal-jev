"""Authenticate every gate row against independently compiled intent inputs."""
import hashlib
import json
import math


def verify_population(artifact, expected, records, policy_id):
    """No torch dependency; expected maps IDs to rederived model inputs.

    The caller authenticates the artifact and expected dataset manifests.
    Matching HF/vLLM rows alone cannot establish that either ran the task.
    """
    cpu={r['id']:r for r in artifact['records']}
    observed={r['id']:r for r in records}
    if (not artifact['passed'] or artifact['requests']!=len(expected)
            or len(cpu)!=len(artifact['records']) or len(observed)!=len(records)
            or set(cpu)!=set(expected) or set(observed)!=set(expected)):
        raise ValueError('complete distinct authenticated population required')
    for uid,request in expected.items():
        source=cpu[uid];row=observed[uid];identity=row['request_identity']
        canonical=json.loads(json.dumps(request))
        if source['request']!=canonical or identity['request']!=canonical:
            raise ValueError('rederived task identity differs: '+uid)
        digest=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if row['request_id']!=digest or identity['policy_id']!=policy_id:
            raise ValueError('request hash or policy differs: '+uid)
        if (identity['version']!='jev-span-request-identity-v1'
                or identity['sampling']!={'temperature':1.0,'policy':'independent_finite_categorical'}
                or identity['audio_processing']!=source['audio_processing']
                or identity['positions']!=source['positions']
                or identity['candidate_ids']!=[m['token_ids'] for m in source['candidate_mapping']]):
            raise ValueError('processor support or audio identity differs: '+uid)
        for name in ('input_ids','input_features','input_features_mask'):
            if identity['inputs'][name]!=source[name]:
                raise ValueError('processor tensor identity differs: '+uid)
        probabilities=row['candidate_logprobs']
        if len(probabilities)!=len(identity['candidate_ids']):raise ValueError('Span coverage differs')
        for values,candidates in zip(probabilities,identity['candidate_ids'],strict=True):
            if (len(values)!=len(candidates) or len(values)<2
                    or any(type(v) not in (float,int) or not math.isfinite(v) or v>0 for v in values)
                    or abs(sum(math.exp(v) for v in values)-1)>1e-6):
                raise ValueError('normalized complete candidate probabilities required')
    return dict(passed=True,requests=len(expected),policy_id=policy_id)
