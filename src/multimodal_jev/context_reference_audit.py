"""Reconstruct reference outputs from complete macro journals and independent inputs.

Caller authenticates runtime/container and input evidence. This verifies the
whole cache, including deduplicated sample mappings, without new inference.
"""
import json
import math
from pathlib import Path

import torch

from .context_logits_wire import validate_logits_record
from .context_reference_dataset import ReferenceDataset,digest
from .span_request_identity import tensor_identity


def audit(root,journal,*,manifest_sha256,identity,population,input_bindings,macro_queries,rollout_queries,replicas):
    dataset=ReferenceDataset(root,manifest_sha256=manifest_sha256,
        expected_identity=identity,expected_population=population)
    journal=Path(journal)
    if set(input_bindings)!=set(population) or any(
            digest(input_bindings[k])!=v or input_bindings[k]['policy_id']!=identity['policy_id']
            for k,v in population.items()):
        raise ValueError('independent complete TRAIN sample/input mapping required')
    if (type(macro_queries) is not int or not 1<=macro_queries<=4096
            or type(rollout_queries) is not int or not 1<=rollout_queries<=macro_queries
            or type(replicas) is not int or not 1<=replicas<=8):
        raise ValueError('declared bounded macro topology required')
    def read(path):
        if path.is_symlink() or not path.resolve().is_relative_to(journal.resolve()):
            raise ValueError('contained regular journal artifact required')
        return json.loads(path.read_text())
    records=json.loads((Path(root)/'records.json').read_text())
    for sample,key in population.items():
        if records[key]['request_identity']!=input_bindings[sample]:
            raise ValueError('cache input differs from independent evidence')
    # Builder visits sorted sample IDs, coalescing only exact duplicate inputs.
    keys=list(dict.fromkeys(population[k] for k in sorted(population)))
    batches=[keys[i:i+macro_queries] for i in range(0,len(keys),macro_queries)]
    expected={f'batch-{i:08d}' for i in range(len(batches))}
    if {p.name for p in journal.glob('batch-*')}!=expected:
        raise ValueError('complete exact reference macro journal required')
    reconstructed={}
    for index,ids in enumerate(batches):
        name=f'batch-{index:08d}';directory=journal/name
        if directory.is_symlink() or (directory/'FAILED.json').exists():
            raise ValueError('failed or aliased reference macro')
        manifest=read(directory/'INPUT.json');complete=read(directory/'COMPLETE.json')
        bindings=[records[k]['request_identity'] for k in ids]
        inputs=[dict(b['inputs'],positions=tensor_identity(torch.tensor([b['positions']],dtype=torch.long))) for b in bindings]
        if (manifest['batch_id']!=name or manifest['policy_id']!=identity['policy_id']
                or manifest['runtime_identity']!=identity['runtime_identity']
                or manifest['inputs']!=inputs or manifest['expected_indices']!=list(range(len(ids)))):
            raise ValueError('macro input order/policy/runtime differs')
        jobs=manifest['jobs'];ordered=[None]*len(ids);hashes={};seen=[]
        for j,job in enumerate(jobs):
            rid=f'rollout-{j:04d}';row=read(directory/(rid+'.json'))
            if (job['rollout_id']!=rid or type(job['replica_id']) is not int
                    or not 0<=job['replica_id']<replicas or not job['indices']
                    or len(job['indices'])>rollout_queries
                    or type(job['estimated_cost']) is not int or job['estimated_cost']<=0
                    or any(type(i) is not int or not 0<=i<len(ids) for i in job['indices'])
                    or any(row[k]!=v for k,v in job.items()) or row['policy_id']!=identity['policy_id']
                    or len(row['records'])!=len(job['indices'])
                    or type(row['seconds']) not in (int,float) or not math.isfinite(row['seconds']) or row['seconds']<=0):
                raise ValueError('complete bounded rollout ownership required')
            hashes[rid]=digest(row)
            for slot,record in zip(job['indices'],row['records'],strict=True):
                validate_logits_record(record)
                cached={k:record[k] for k in ('request_id','request_identity','prob','policy_id')}
                if ordered[slot] is not None or cached!=records[ids[slot]]:
                    raise ValueError('reference cache differs from actual unique inference output')
                ordered[slot]=record;seen.append(slot);reconstructed[ids[slot]]=cached
        if sorted(seen)!=list(range(len(ids))):raise ValueError('complete macro coverage required')
        if (complete['batch_id']!=name or complete['policy_id']!=identity['policy_id']
                or complete['status']!='complete_macro_inference' or complete['actor_update_performed'] is not False
                or complete['input_sha256']!=digest(manifest) or complete['rollout_hashes']!=hashes
                or complete['completed_indices']!=list(range(len(ids))) or complete['result_sha256']!=digest(ordered)
                or type(complete['seconds']) not in (int,float) or not math.isfinite(complete['seconds']) or complete['seconds']<=0):
            raise ValueError('complete reference macro receipt differs')
    if reconstructed!=records:raise ValueError('complete reference result reconstruction required')
    return dict(status='reference_dataset_journal_audited',recordings=len(dataset.population),
        distinct_inputs=len(keys),macro_batches=len(batches),manifest_sha256=manifest_sha256,
        policy_id=identity['policy_id'],numerical_parity_verified=False,
        scope='Cache outputs exactly reconstructed from journals and independent input identities; runtime admission separate')
