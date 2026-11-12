#!/usr/bin/env python3
"""Independent tensor and backend audit of a completed resident two-step replay.

Artifacts only. The host controller must separately observe actual Docker exits.
"""
from pathlib import Path
import torch
from multimodal_jev.baseline import file_hash
from multimodal_jev.policy_identity import checkpoint_identity
from multimodal_jev.intent_resident_replay import compare_state
from queue_span_native_actor_npu import read
from queue_span_dp4_resume import check_pins
from queue_intent_finite_npu import verify_final,verify_gate
from probe_intent_resident_replay_npu import validate


def verify(root,suite):
    root,suite=map(Path,(root,suite));plan=read(root/'config.json');configs=validate(plan)
    cfg=read(suite/'config.json');launch=read(root/'launch.json')
    check_pins({str(root/'config.json'):launch['config_sha256'],str(root/'source-manifest.json'):launch['source_manifest_sha256']})
    check_pins({str(root/'source'/p):h for p,h in read(root/'source-manifest.json').items()});check_pins(cfg['pins'])
    if (cfg['resident_training_root']!=str(root) or cfg['adapter']!=str(root/'run/resident-adapter')
            or cfg['population']!={'recordings':89,'requests':178}):raise ValueError('exact resident final export/full gate required')
    refs=[Path(e['training']) for e in plan['references']];suites=[Path(e['suite']) for e in plan['references']]
    evidence={};ranks=[]
    for training,gate in zip(refs,suites,strict=True):
        verify_final(gate);v=read(gate/'verified.json')
        if v['root']!=str(training):raise ValueError('reference acceptance lineage differs')
        evidence.update(v['files'])
    gate=verify_gate(suite,configs[0]['pilot'],configs[0]['manifest_sha256'])
    ids=[]
    for step,adapter in enumerate((root/'run/adapter',root/'run/resident-adapter')):
        identity,files=checkpoint_identity(adapter)
        if identity!=checkpoint_identity(refs[step]/'run/adapter')[0]:raise ValueError('resident exported adapter differs')
        evidence.update({str(adapter/n):h for n,h in files.items()});ids.append(identity)
    if cfg['adapter_id']!=ids[1] or cfg['base_id']!=read(suites[1]/'config.json')['base_id']:raise ValueError('final gate policy differs')
    for name,reference in [('selection.json',refs[0]/'run/selection.json'),('supervision.json',refs[0]/'run/supervision.json'),('resident-selection.json',refs[1]/'run/selection.json')]:
        if read(root/'run'/name)!=read(reference):raise ValueError('resident data schedule or supervision differs')
    for rank in range(4):
        first=root/'run'/f'rank-{rank}';second=root/'run'/f'resident-rank-{rank}'
        if read(first/'terminal.json')['exit_code']!=0:raise ValueError('resident rank did not complete')
        final=read(second/'result.json')
        if (final['passed'] is not True or final['rank']!=rank or final['updates']!=2
                or final['phase']!='awaiting_acceptance' or final['resident_objects_preserved'] is not True
                or final['policy_id']!=cfg['base_id']+':'+ids[1]):raise ValueError('resident final phase/policy differs')
        states=[]
        for index,out in enumerate((first,second)):
            ref=refs[index]/'run'/f'rank-{rank}'
            states.append(compare_state(torch.load(ref/'optimizer-local.pt',map_location='cpu',weights_only=True),
                                        torch.load(out/'optimizer-local.pt',map_location='cpu',weights_only=True),rank=rank,updates=index+1))
            for name in (('before.json','after.json') if index==0 else ('before.json',)):
                if read(out/name)!=read(ref/name):raise ValueError('resident exact actor/behavior probability differs')
        ranks.append(dict(rank=rank,states=states))
    # Fresh backend results must match the historical same-policy full population.
    for backend in ('hf','vllm'):
        a=read(suites[1]/f'run/{backend}/result.json')['records'];b=read(suite/f'run/{backend}/result.json')['records']
        aa={r['id']:r for r in a};bb={r['id']:r for r in b}
        if len(a)!=178 or len(b)!=178 or len(aa)!=178 or aa.keys()!=bb.keys():raise ValueError('full reference/backend population required')
        for uid in aa:
            for field in ('request_id','request_identity','candidate_logprobs'):
                if aa[uid][field]!=bb[uid][field]:raise ValueError('fresh replay backend differs from same-policy reference')
    for base in (root/'run',suite/'run'):
        for p in base.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.pt','.safetensors'):evidence[str(p)]=file_hash(p)
    for p in (root/'config.json',root/'source-manifest.json',root/'launch.json',suite/'config.json',suite/'source-manifest.json',suite/'launch.json',Path(__file__),Path(__file__).resolve().parents[2]/'src/multimodal_jev/intent_resident_replay.py'):
        evidence[str(p)]=file_hash(p)
    return dict(passed=True,updates=2,ranks=ranks,adapter_id=ids[1],full_gate=gate,files=evidence,
        scope='Two resident FSDP actor updates exactly match accepted D1/D2 FP32 master/Adam and BF16 exports; fresh178 HF/vLLM results exact. Historical accepted behaviors reused. No resident rollout or complete continuous trainer proof.')
