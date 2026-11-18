"""Freeze an input-only TRAIN panel for endpoint HF/replica numerical checks.

Selection uses both length axes and deterministic ID hashes, never targets,
scores or model failures. This prepares inputs, not numerical admission.
"""
import argparse
import copy
import json
from pathlib import Path
from types import FunctionType

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_candidate_task import compile_candidates
from multimodal_jev.context_decision_contract import ContextDecisionRequest,ApplicabilityQuestion
from multimodal_jev.context_reference_dataset import digest
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.readonly_encoded_audio_cache import ReadOnlyEncodedAudioCache
from multimodal_jev.slurp_portable_release import load as load_source
from multimodal_jev.slurp_training import content_hash
from tools.jev.context_cache_receipts import parse_stream
from tools.reports.context_reference_input_evidence import load as load_evidence
from tools.reports.context_selection_audited_cache import encoder as shared_encoder


RULE=dict(schema='jev-train-parity-selection-v1',minimum_queries=256,
    extremes_per_axis=16,axes=['input_ids','input_features'],
    ordering='sample_id',fill='sha256_json_sample_id',labels_used=False)


def select(evidence):
    """At least 256 inputs, retaining all 16 extrema on both length axes."""
    if len(evidence)<RULE['minimum_queries']:raise ValueError('complete source population required')
    lengths={}
    for sample,row in evidence.items():
        shapes=[row['tensors'][axis]['shape'] for axis in RULE['axes']]
        text,audio=shapes
        if (not isinstance(sample,str) or not sample or len(text)!=2 or text[0]!=1
                or type(text[-1])is not int or not 0<text[-1]<=4096
                or len(audio)!=3 or audio[0]!=1 or audio[1]!=128
                or type(audio[-1])is not int or audio[-1]<=0):
            raise ValueError('bounded source text and acoustic lengths required')
        lengths[sample]=(text[-1],audio[-1])
    selected=set();extremes={}
    for i,axis in enumerate(RULE['axes']):
        ordered=sorted(lengths,key=lambda k:(lengths[k][i],k))
        extremes[axis]=dict(shortest=ordered[:16],longest=ordered[-16:])
        selected.update(extremes[axis]['shortest']);selected.update(extremes[axis]['longest'])
    for sample in sorted(lengths,key=lambda k:(content_hash(k),k)):
        if len(selected)>=256:break
        selected.add(sample)
    ids=sorted(selected)
    return dict(rule=copy.deepcopy(RULE),ids=ids,extremes=extremes,
        population_sha256=digest(evidence),population=len(evidence),
        lengths={sample:list(lengths[sample]) for sample in ids})


def reconstruct(c):
    rows,task,_=load_source(c['release'],c['release_sha256'],'learning',purpose='optimize')
    evidence,proof=load_evidence(c,rows=rows,task=task,release_sha256=c['release_sha256'],
        processor_sha256=c['processor_sha256'])
    if len(rows)!=40513:raise ValueError('full clean TRAIN source required')
    selected=select(evidence);by_id={row['id']:row for row in rows}
    stream,_=parse_stream(Path(c['input_receipts']).read_text())
    receipts={r['value']['id']:r['value'] for r in stream[1:-1]}
    candidates=[dict(id=v['id'],definition=v['description']) for v in task['schema']['candidates']]
    inputs=[]
    for sample in selected['ids']:
        row=by_id[sample];receipt=receipts[sample]
        request=compile_candidates(row['audio'],candidates,task['thinking']['requirements'],semantics='exclusive_event')
        inputs.append(dict(id=sample,request=json.loads(json.dumps(request.model_input())),
            receipt={k:receipt[k] for k in ('id','audio_sha256','request_sha256','cache_key','cache_sha256','tensors')}))
    return dict(panel=selected,inputs=inputs,candidate_ids=[v['id'] for v in candidates],
        source=dict(release_sha256=c['release_sha256'],processor_sha256=c['processor_sha256'],
            input_proof_sha256=c['input_proof_sha256'],input_receipts_sha256=c['input_receipts_sha256']),
        input_proof=proof)


def prepare(c,provenance):
    values=reconstruct(c);output=Path(c['output'])
    sources=[Path(c[k]) for k in ('release','input_proof','input_receipts')]
    if output.exists() or any(output.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(output.resolve()) for p in sources):
        raise ValueError('fresh independent panel output required')
    check_paths({'train_parity_inputs':output});output.mkdir(parents=True,exist_ok=False)
    commit_json(output/'inputs.json',values)
    manifest=dict(schema='jev-train-parity-inputs-v1',files={'inputs.json':file_hash(output/'inputs.json')},
        source=values['source'],recordings=len(values['inputs']),provenance=provenance,
        labels_used=False,cache_bytes_verified=False,numerical_admission=False)
    commit_json(output/'MANIFEST.json',manifest)
    return dict(status='train_parity_inputs_prepared_only',manifest_sha256=file_hash(output/'MANIFEST.json'),
        recordings=manifest['recordings'],population=values['panel']['population'],npu_executed=False)


def load(c):
    root=Path(c['output']);manifest=root/'MANIFEST.json'
    if root.is_symlink() or manifest.is_symlink() or file_hash(manifest)!=c['manifest_sha256']:
        raise ValueError('train parity manifest differs')
    m=json.loads(manifest.read_text())
    if (m['schema']!='jev-train-parity-inputs-v1' or set(m['files'])!={'inputs.json'}
            or any(m[k] is not False for k in ('labels_used','cache_bytes_verified','numerical_admission'))
            or any(p.is_symlink() for p in root.rglob('*'))
            or {str(p.relative_to(root)):file_hash(p) for p in root.rglob('*') if p.is_file() and p!=manifest}!=m['files']):
        raise ValueError('complete immutable input-only panel required')
    values=json.loads((root/'inputs.json').read_text())
    if (values['source']!=m['source'] or len(values['inputs'])!=m['recordings']
            or m['recordings']!=RULE['minimum_queries']
            or values['panel']['rule']!=RULE or values['panel']['population']!=40513
            or [r['id'] for r in values['inputs']]!=values['panel']['ids']
            or values['panel']['ids']!=sorted(set(values['panel']['ids']))
            or len(values['candidate_ids'])!=60 or len(set(values['candidate_ids']))!=60):
        raise ValueError('complete native60 TRAIN panel binding required')
    for row in values['inputs']:
        if (set(row)!={'id','request','receipt'}
                or set(row['receipt'])!={'id','audio_sha256','request_sha256','cache_key','cache_sha256','tensors'}
                or row['receipt']['id']!=row['id']
                or set(row['request'])!={'audio','questions','requirements','history','version'}):
            raise ValueError('only label-free request and cache evidence fields allowed')
    return values


def audit(c,provenance):
    actual=load(c);expected=reconstruct(c)
    if actual!=expected:raise ValueError('panel differs from full TRAIN independent reconstruction')
    return dict(status='train_parity_inputs_independently_audited',manifest_sha256=c['manifest_sha256'],
        recordings=len(actual['inputs']),population=40513,panel_sha256=digest(actual['panel']),
        labels_used=False,cache_bytes_verified=False,numerical_admission=False,provenance=provenance)


def encoder(c,*,processor_sha256,decision_token_id,audio_token_id):
    values=load(c)
    if processor_sha256!=values['source']['processor_sha256']:raise ValueError('source processor differs')
    requests={};receipts=[];assets=[]
    for row in values['inputs']:
        raw=row['request'];prior=row['receipt']
        if (set(row)!={'id','request','receipt'} or prior['id']!=row['id']
                or set(raw)!={'audio','questions','requirements','history','version'}
                or raw['version']!='jev-context-exclusive_event-v1'
                or raw['history'] or content_hash(raw)!=prior['request_sha256']
                or [q['id'] for q in raw['questions']]!=values['candidate_ids']):
            raise ValueError('label-free source request identity differs')
        request=ContextDecisionRequest(raw['audio'],tuple(ApplicabilityQuestion(**q) for q in raw['questions']),
            tuple(raw['requirements']),tuple(raw['history']),version=raw['version'])
        requests[row['id']]=request;receipts.append(prior)
        assets.append({k:prior[k] for k in ('id','audio_sha256','request_sha256')})
    readonly_encoder=FunctionType(shared_encoder.__code__,
        dict(shared_encoder.__globals__,EncodedAudioCache=ReadOnlyEncodedAudioCache),'train_readonly_encoder')
    encode=readonly_encoder(c['cache_root'],requests,values['candidate_ids'],receipts,
        processor_sha256=processor_sha256,decision_token_id=decision_token_id,audio_token_id=audio_token_id)
    return assets,encode


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['prepare','audit'],default='prepare')
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    result=prepare(c,provenance) if a.action=='prepare' else audit(c,provenance)
    if a.action=='audit':commit_json(c['audit_output'],result)
    print(json.dumps(result,indent=2))
