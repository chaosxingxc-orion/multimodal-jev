#!/usr/bin/env python3
"""Candidate full native FSC inference from frozen source-selected weights.

Consumes only label-free requests and already audited processor caches. No
target scoring, teacher calls, decoding, optimizer or target-model selection.
Actual 31-span HF/vLLM parity and launch admission remain separate obligations.
"""
import argparse
import json
from pathlib import Path
import time

from multimodal_jev.baseline import file_hash
from multimodal_jev.slurp_training import content_hash
from context_cache_receipts import parse_stream


def read_bound(entry):
    if file_hash(entry['path'])!=entry['sha256']:
        raise ValueError('bound artifact changed')
    return json.loads(Path(entry['path']).read_text())


def load_inputs(config):
    """Validate the entire input package without loading target labels/weights."""
    package=read_bound(config['input_manifest'])
    proof=read_bound(config['input_audit'])
    requests_path=Path(config['requests'])
    if file_hash(requests_path)!=package['files']['requests.jsonl']:
        raise ValueError('input requests changed')
    if (proof['state']['Status']!='exited' or proof['state']['ExitCode']!=0 or
            proof['state']['OOMKilled'] or proof['manifest']['recordings']!=3793 or
            proof['manifest']['decisions']!=31 or
            proof['manifest']['request_manifest_sha256']!=config['input_manifest']['sha256'] or
            file_hash(config['input_receipts'])!=proof['independent_receipts_sha256']):
        raise ValueError('full independently audited FSC inputs required')
    stream,_=parse_stream(Path(config['input_receipts']).read_text())
    receipts=[row['value'] for row in stream[1:-1]]
    rows=[json.loads(line) for line in requests_path.read_text().splitlines()]
    if (len(rows)!=3793 or len(receipts)!=3793 or len({r['id'] for r in rows})!=3793
            or package['recordings']!=3793 or len(package['candidate_ids'])!=31
            or len(set(package['candidate_ids']))!=31):
        raise ValueError('complete unique native31 population required')
    for row,receipt in zip(rows,receipts,strict=True):
        if set(row)!={'id','audio_sha256','speaker_id','request','request_sha256','rendered_sha256'}:
            raise ValueError('label-free input fields required')
        raw=row['request']
        if (set(raw)!={'audio','questions','requirements','history','version'} or
                raw['version']!='jev-context-exclusive_event-v1' or
                len(raw['questions'])!=31 or
                content_hash(raw)!=row['request_sha256'] or
                any(row[key]!=receipt[key] for key in ('id','audio_sha256','request_sha256'))):
            raise ValueError('request/processor binding differs')
    return rows,receipts,proof


def validate_source(config):
    selection=read_bound(config['source_selection'])
    role=config['source_role']
    if not selection['complete'] or selection['selected_checkpoint'] is None:
        raise ValueError('complete source-only selection required')
    if role not in ('initial','selected'):
        raise ValueError('preselected source pair only')
    expected=(selection['initial_policy_id'] if role=='initial'
              else selection['selected_checkpoint']['policy_id'])
    if config['policy_id']!=expected:
        raise ValueError('target inference policy not selected by source curve')


def checked_prediction(row,receipt,prediction,policy_id):
    identity=dict(policy_id=policy_id,positions=receipt['positions'][0],
                  inputs={k:v for k,v in receipt['tensors'].items() if k!='positions'})
    import math
    if (prediction['policy_id']!=policy_id or prediction['request_identity']!=identity or
            prediction['request_id']!=content_hash(identity) or len(prediction['prob'])!=31 or
            any(type(p) not in (float,int) or not math.isfinite(p) or not 0<=p<=1
                for p in prediction['prob'])):
        raise ValueError('engine policy/input/probability differs')
    return dict(prediction,id=row['id'])


def audited_encoder(config,proof):
    from multimodal_jev.cached_context_preparation import CachedContextPreparer,processor_cache_identity
    from multimodal_jev.context_decision_contract import ApplicabilityQuestion,ContextDecisionRequest
    from multimodal_jev.encoded_audio_cache import EncodedAudioCache,key_for
    from multimodal_jev.span_request_identity import tensor_identity
    processor_id=processor_cache_identity(config['snapshot'])
    if processor_id!=proof['manifest']['processor_sha256']:
        raise ValueError('source processor differs from audited inputs')
    cache_root=Path(config['cache_root'])
    if not cache_root.is_dir():raise ValueError('existing audited cache required')
    def no_rebuild():raise ValueError('audited cache miss; do not rebuild during transfer')
    prepare=CachedContextPreparer(no_rebuild,EncodedAudioCache(cache_root),
        processor_sha256=processor_id,decision_token_id=151706,audio_token_id=151676,max_tokens=4096)
    def encode(pair):
        row,receipt=pair;raw=row['request']
        request=ContextDecisionRequest(raw['audio'],tuple(ApplicabilityQuestion(**q) for q in raw['questions']),
            tuple(raw['requirements']),tuple(raw['history']),version=raw['version'])
        key=key_for(prepare.identity(request,row['audio_sha256']))
        if key!=receipt['cache_key'] or file_hash(cache_root/(key+'.pt'))!=receipt['cache_sha256']:
            raise ValueError('audited cache bytes changed')
        value=prepare((request,row['audio_sha256']))
        tensors={k:tensor_identity(v) for k,v in dict(value.inputs,positions=value.positions).items()}
        if tensors!=receipt['tensors'] or value.positions.tolist()!=receipt['positions']:
            raise ValueError('actual encoded inputs differ from audit')
        return row,receipt,value
    return encode


def run(config,out):
    import torch
    import torch_npu
    from multimodal_jev.context_policy_artifacts import read_execution_bundle,verify_native_adapter
    from multimodal_jev.context_runtime_profile import PROFILE
    from multimodal_jev.context_resident_rollout import FixedContextRollout
    from multimodal_jev.local_runtime_storage import require_training_storage
    from multimodal_jev.local_checkpoint import commit_json
    from multimodal_jev.ordered_prefetch import OrderedPrefetch
    from multimodal_jev.baseline import atomic_json
    validate_source(config)
    rows,receipts,proof=load_inputs(config)
    require_training_storage(out)
    for path,digest in config['pins'].items():
        if file_hash(path)!=digest:raise ValueError('runtime dependency changed')
    if (len(config['physical_devices'])!=1 or
            sorted(p.name for p in Path('/dev').glob('davinci[0-9]*'))!=
            ['davinci'+str(config['physical_devices'][0])]):
        raise ValueError('one isolated NPU required')
    torch.set_num_threads(2);torch.npu.set_device(0)
    bundle=read_execution_bundle(config['snapshot'],expected_policy_id=config['policy_id'],
        expected_contract=config['contract'],expected_base_id=config['base_id'])
    if (bundle['contract']['runtime_profile']!=PROFILE or
            bundle['contract']['request_version']!='jev-context-exclusive_event-v1' or
            bundle['contract']['token']['registration_status']!='admitted'):
        raise ValueError('admitted runtime and task contract required')
    verify_native_adapter(config['adapter'],bundle)
    encode=audited_encoder(config,proof)
    engine_config={key:config[key] for key in
        ('model','base_id','snapshot','policy_id','contract','adapter','physical_devices')}
    engine_config.update(tensor_parallel_size=1,max_model_len=4096,max_decisions=31)
    engine=FixedContextRollout(engine_config);records=[];started=time.monotonic()
    try:
        commit_json(out/'runtime.json',engine.runtime)
        with OrderedPrefetch(encode,list(zip(rows,receipts)),workers=1,ahead=2) as prepared:
            for index,(row,receipt,value) in enumerate(prepared):
                prediction=checked_prediction(row,receipt,engine.score(value.inputs,value.positions,
                    audio_token_id=151676),config['policy_id'])
                commit_json(out/f'record-{index:05d}.json',dict(prediction=prediction,input_receipt=receipt))
                records.append(prediction)
                if (index+1)%64==0:
                    atomic_json(out/'progress.json',dict(completed=index+1,total=3793,seconds=time.monotonic()-started))
    finally:engine.close()
    if len(records)!=3793:raise ValueError('incomplete transfer population')
    commit_json(out/'predictions.json',dict(records=records))
    report=dict(recordings=3793,decisions=31,policy_id=config['policy_id'],
        source_role=config['source_role'],predictions_sha256=file_hash(out/'predictions.json'),
        seconds=time.monotonic()-started,scope='Label-free full FSC inference; independent collection and scoring still required')
    commit_json(out/'result.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    run(json.loads(args.config.read_text()),args.output)
