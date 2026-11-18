"""Convert native backend evidence only after exact input/policy binding checks."""
import hashlib
import json


def from_backend(artifact, result, *, policy_id):
    if (artifact.get('passed') is not True or artifact.get('population_mode') != 'full_selection'
            or result.get('passed') is not True or result.get('backend') not in ('hf', 'vllm')
            or result.get('base_id', '') + ':' + result.get('adapter_id', '') != policy_id):
        raise ValueError('successful full selection inputs and exact backend policy required')
    inputs = {row['id']: row for row in artifact['records']}
    if len(inputs) != artifact['requests'] or artifact['recordings'] != artifact['requests']:
        raise ValueError('complete unique input population required')
    seen = set(); predictions = []
    for output in result['records']:
        rid = output['id']
        if rid not in inputs or rid in seen:
            raise ValueError('duplicate or foreign backend output')
        seen.add(rid); row = inputs[rid]; identity = output['request_identity']
        if (rid != row['recording_id'] or row['view'] != 'direct' or row['variant'] != 'native'
                or len(row['candidate_mapping']) != 1 or len(output['candidate_logprobs']) != 1
                or identity['policy_id'] != policy_id or identity['request'] != row['request']
                or identity['audio_processing'] != row['audio_processing']
                or identity['positions'] != row['positions']
                or identity['candidate_ids'] != [m['token_ids'] for m in row['candidate_mapping']]
                or any(identity['inputs'][key] != row[key] for key in
                       ('input_ids', 'input_features', 'input_features_mask'))
                or hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'),
                                             allow_nan=False).encode()).hexdigest() != output['request_id']):
            raise ValueError('backend request differs from independently prepared input')
        mapping = row['candidate_mapping'][0]
        values = mapping['values']; logp = output['candidate_logprobs'][0]
        if len(values) != 60 or len(logp) != 60:
            raise ValueError('complete native60 distribution required')
        predictions.append(dict(id=rid, audio_sha256=row['audio_sha256'], policy_id=policy_id,
            view='direct', variant='native', status='completed', candidate_values=values,
            candidate_logprobs=logp, prediction=values[max(range(60), key=lambda i: logp[i])],
            request_id=output['request_id'], backend=result['backend']))
    if seen != inputs.keys():
        raise ValueError('partial backend run cannot produce full metrics')
    return predictions
