"""Bind full TRAIN reference inputs directly to the archived CPU audit stream.

Original independent hashes authenticate marker values read from a new cache;
the producer's own record is never accepted as independent evidence by itself.
"""
import json
from pathlib import Path

import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_candidate_task import compile_candidates
from multimodal_jev.context_reference_dataset import digest
from multimodal_jev.slurp_training import content_hash
from multimodal_jev.span_request_identity import tensor_identity
from tools.jev.context_cache_receipts import parse_stream


def bind(proof,stream,*,rows,task,release_sha256,processor_sha256):
    manifest=proof['manifest'];state=proof['state']
    entries,diagnostics=parse_stream(stream)
    if (state['Status']!='exited' or state['ExitCode']!=0 or state['OOMKilled']
            or manifest['schema']!='context-full-learning-cache-v1'
            or manifest['recordings']!=40513 or len(rows)!=40513
            or manifest['release_sha256']!=release_sha256
            or manifest['processor_sha256']!=processor_sha256
            or manifest['request_version']!='jev-context-exclusive_event-v1'
            or entries[-1]['value']!=manifest or entries[0]['value']['recordings']!=40513):
        raise ValueError('complete successful independent TRAIN input evidence required')
    receipts={r['value']['id']:r['value'] for r in entries[1:-1]}
    if len(receipts)!=len(rows) or set(receipts)!={r['id'] for r in rows}:
        raise ValueError('complete unique TRAIN sample identity required')
    candidates=[dict(id=v['id'],definition=v['description']) for v in task['schema']['candidates']]
    if len(candidates)!=60 or len({v['id'] for v in candidates})!=60:
        raise ValueError('complete native60 ontology required')
    result={}
    for row in rows:
        r=receipts[row['id']]
        request=compile_candidates(row['audio'],candidates,task['thinking']['requirements'],semantics='exclusive_event')
        tensors=r['tensors']
        if (r['audio_sha256']!=row['audio_sha256'] or r['request_sha256']!=content_hash(request.model_input())
                or set(tensors)!={'input_ids','attention_mask','input_features','input_features_mask','positions'}
                or tensors['positions']['shape']!=[1,60] or tensors['positions']['dtype']!='torch.int64'
                or tensors['input_features']['dtype']!='torch.bfloat16'):
            raise ValueError('TRAIN request/audio/processor tensors differ from independent evidence')
        result[row['id']]=dict(tensors=tensors,audio_sha256=r['audio_sha256'],request_sha256=r['request_sha256'])
    return result,dict(recordings=len(result),candidate_ids=[v['id'] for v in candidates],
        release_sha256=release_sha256,processor_sha256=processor_sha256,driver_diagnostics=diagnostics,
        scope='Full TRAIN requests bound to original independent tensor hashes; no new inference or runtime admission')


def bindings_from_records(evidence,records,population,*,policy_id):
    if set(evidence)!=set(population) or set(records)!=set(population.values()):
        raise ValueError('complete reference and independent TRAIN population required')
    bindings={}
    for sample,key in population.items():
        record=records[key];binding=record['request_identity']
        if (record['request_id']!=key or record['policy_id']!=policy_id
                or binding['policy_id']!=policy_id or digest(binding)!=key):
            raise ValueError('reference input/policy identity differs')
        positions=binding['positions']
        if (len(positions)!=60 or any(type(p)is not int or p<0 for p in positions)
                or positions!=sorted(set(positions))):
            raise ValueError('complete ordered integer marker positions required')
        actual=dict(binding['inputs'],positions=tensor_identity(torch.tensor([positions],dtype=torch.long)))
        if actual!=evidence[sample]['tensors']:
            raise ValueError('reference input tensor or marker differs from independent hash')
        bindings[sample]=binding
    return bindings


class AuditedPreparer:
    """Validate actual cached tensor content before it reaches an inference macro."""
    def __init__(self,preparer,evidence):
        self.preparer=preparer;self.expected={}
        for row in evidence.values():
            key=(row['request_sha256'],row['audio_sha256'])
            if key in self.expected and self.expected[key]!=row['tensors']:
                raise ValueError('conflicting independent receipts for identical request/audio')
            self.expected[key]=row['tensors']

    def __call__(self,job):
        request,audio_sha256=job
        key=(content_hash(request.model_input()),audio_sha256)
        if key not in self.expected:raise ValueError('reference input absent from independent TRAIN evidence')
        value=self.preparer(job)
        actual={k:tensor_identity(v) for k,v in dict(value.inputs,positions=value.positions).items()}
        if actual!=self.expected[key]:
            raise ValueError('actual reference input differs from independently audited cached tensors')
        return value


def load(c,*,rows,task,release_sha256,processor_sha256):
    for name in ('input_proof','input_receipts'):
        if file_hash(c[name])!=c[name+'_sha256']:raise ValueError('independent input evidence changed: '+name)
    proof=json.loads(Path(c['input_proof']).read_text())
    if proof['independent_receipts_sha256']!=c['input_receipts_sha256']:
        raise ValueError('input proof does not authenticate the supplied stream')
    return bind(proof,Path(c['input_receipts']).read_text(),rows=rows,task=task,
                release_sha256=release_sha256,processor_sha256=processor_sha256)


def run(c,provenance):
    from multimodal_jev.local_checkpoint import commit_json
    from multimodal_jev.slurp_portable_release import load as load_source
    rows,task,_=load_source(c['release'],c['release_sha256'],'learning',purpose='optimize')
    evidence,proof=load(c,rows=rows,task=task,release_sha256=c['release_sha256'],processor_sha256=c['processor_sha256'])
    proof.update(status='full_train_independent_input_stream_bound',
        input_proof_sha256=c['input_proof_sha256'],input_receipts_sha256=c['input_receipts_sha256'],
        input_population_sha256=digest(evidence),provenance=provenance,
        reference_outputs_built=False,numerical_parity_verified=False)
    commit_json(c['output'],proof)
    return proof


if __name__=='__main__':
    import argparse
    from multimodal_jev.local_config import load_config
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    print(json.dumps(run(c,provenance),indent=2))
