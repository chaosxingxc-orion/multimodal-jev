"""Immutable reference results, loaded once before training.

Publication identity and producer numerical admission are caller-owned. This
module proves content/coverage and exact request binding, never NPU equivalence.
No online inference fallback, mutable entries, labels or behavior cache.
"""
import copy
import hashlib
import json
import math
from pathlib import Path

import torch

from .local_checkpoint import commit_json
from .local_runtime_storage import check_paths
from .span_request_identity import tensor_identity


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_identity(identity):
    required = {'schema', 'policy_id', 'base_id', 'contract', 'runtime_identity',
                'release_sha256', 'candidate_ids', 'population_sha256'}
    if (set(identity) != required or identity['schema'] != 'jev-reference-dataset-v1'
            or identity['contract']['base_id'] != identity['base_id']):
        raise ValueError('complete reference dataset identity required')
    runtime = identity['runtime_identity']
    if (set(runtime) != {'image_sha256','source_manifest_sha256','processor_sha256','profile','output_protocol'}
            or not isinstance(runtime['profile'],str) or not runtime['profile']
            or not isinstance(runtime['output_protocol'],str) or not runtime['output_protocol']):
        raise ValueError('complete producer runtime required')
    for value in [identity[k] for k in ('policy_id','base_id','release_sha256','population_sha256')] + [
            runtime[k] for k in ('image_sha256','source_manifest_sha256','processor_sha256')]:
        if type(value) is not str or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('SHA256 identities required')
    ids=identity['candidate_ids']
    if not isinstance(ids,list) or len(ids)<2 or any(type(x)is not str or not x for x in ids) or len(set(ids))!=len(ids):
        raise ValueError('ordered unique candidates required')


def validate_record(record, identity):
    if (set(record) != {'request_id','request_identity','prob','policy_id'}
            or record['policy_id'] != identity['policy_id']):
        raise ValueError('only reference probability records accepted')
    request=record['request_identity'];n=len(identity['candidate_ids'])
    if (set(request) != {'policy_id','inputs','positions'}
            or request['policy_id'] != identity['policy_id']
            or digest(request) != record['request_id']):
        raise ValueError('reference request binding differs')
    positions=request['positions']
    if (len(positions)!=n or any(type(v)is not int or v<0 for v in positions)
            or positions!=sorted(set(positions))):
        raise ValueError('complete ordered candidate marker positions required')
    inputs=request['inputs']
    if (not {'input_ids','attention_mask','input_features','input_features_mask'} <= set(inputs)
            or set(inputs)-{'input_ids','attention_mask','input_features','input_features_mask','position_ids'}):
        raise ValueError('complete label-free input identities required')
    if positions[-1]>=inputs['input_ids']['shape'][1]:
        raise ValueError('marker outside input')
    p=record['prob']
    if (len(p)!=n or any(type(v)not in (float,int) or not math.isfinite(v) or not 0<v<1 for v in p)):
        raise ValueError('finite interior reference probabilities required')
    values=torch.tensor(p,dtype=torch.float32)
    if ((values<=0)|(values>=1)).any() or values.tolist()!=p:
        raise ValueError('exact FP32 probability representation required')


def publish(root, *, identity, records, population, producer):
    """population maps source sample IDs to complete request IDs, not labels."""
    validate_identity(identity)
    root=Path(root);check_paths({'reference_dataset':root})
    if (not population or any(type(k)is not str or not k or type(v)is not str for k,v in population.items())
            or identity['population_sha256']!=digest(population)):
        raise ValueError('complete authenticated sample-to-input map required')
    by_request={}
    for record in records:
        validate_record(record,identity);key=record['request_id']
        if key in by_request:raise ValueError('duplicate reference input')
        by_request[key]=record
    if set(by_request)!=set(population.values()):
        raise ValueError('reference results must cover exactly the declared population')
    root.mkdir(parents=True,exist_ok=False)
    # Contract/runtime are stored once, not repeated for all 40,513 samples.
    commit_json(root/'records.json',by_request)
    commit_json(root/'population.json',population)
    manifest=dict(identity=identity,files={n:sha(root/n) for n in ('records.json','population.json')},
        recordings=len(population),distinct_inputs=len(by_request),producer=producer,
        numerical_admission_claim=False)
    commit_json(root/'MANIFEST.json',manifest)
    return dict(manifest_sha256=sha(root/'MANIFEST.json'),recordings=len(population),
                distinct_inputs=len(by_request))


class ReferenceDataset:
    """Read-only in-memory table. Missing/stale input fails before any rollout."""
    def __init__(self,root,*,manifest_sha256,expected_identity,expected_population):
        root=Path(root)
        names=('MANIFEST.json','records.json','population.json')
        if any((root/n).is_symlink() for n in names) or sha(root/'MANIFEST.json')!=manifest_sha256:
            raise ValueError('pinned regular reference publication required')
        manifest=json.loads((root/'MANIFEST.json').read_text())
        validate_identity(expected_identity)
        if (manifest['identity']!=expected_identity
                or set(manifest['files'])!=set(names[1:])
                or any(sha(root/n)!=h for n,h in manifest['files'].items())):
            raise ValueError('reference publication identity or content differs')
        population=json.loads((root/'population.json').read_text())
        if (population!=expected_population or digest(population)!=expected_identity['population_sha256']
                or manifest['recordings']!=len(population)):
            raise ValueError('training population differs')
        self._records=json.loads((root/'records.json').read_text())
        if set(self._records)!=set(population.values()) or manifest['distinct_inputs']!=len(self._records):
            raise ValueError('incomplete reference dataset')
        for key,record in self._records.items():
            validate_record(record,expected_identity)
            if key!=record['request_id']:raise ValueError('reference index differs')
        self.identity=copy.deepcopy(expected_identity)
        self.policy_id=expected_identity['policy_id']
        self.runtime_identity=copy.deepcopy(expected_identity['runtime_identity'])
        self.manifest_sha256=manifest_sha256
        self.population=dict(population)

    digest=staticmethod(digest)

    def binding(self,inputs,positions):
        return dict(policy_id=self.policy_id,positions=positions[0].tolist(),
            inputs={k:tensor_identity(v) for k,v in sorted(inputs.items())})

    def read(self,inputs,positions):
        identity=self.binding(inputs,positions)
        record=self._records.get(digest(identity))
        if record is None or record['request_identity']!=identity:
            raise ValueError('reference dataset miss: rebuild before training')
        return copy.deepcopy(record)

    def write(self,*args,**kwargs):
        raise ValueError('reference dataset is read-only; publish a new version')
