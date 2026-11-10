#!/usr/bin/env python3
"""Independently accept the fixed pretrained TRAIN6 CPU forward probe."""
import argparse
import json
import math
from pathlib import Path

from multimodal_jev.baseline import atomic_json,file_hash
from sample_sfc_omni_learning import load_inputs
from watch_qwen3_omni_probe import inspect


def read(path):return json.loads(Path(path).read_text())


def check_container(state,launch,intent):
    host=state['HostConfig']
    if (state['Id']!=launch['container_id'] or state['Image']!=launch['image'] or launch['image']!=intent['image']
            or state['State']['Status']!='exited' or state['State']['ExitCode']!=0
            or state['State'].get('OOMKilled',False) or host.get('Devices') or host.get('Privileged')
            or host['NetworkMode']!='none' or host['Memory']!=160*1024**3 or host['NanoCpus']!=16000000000
            or intent['wall_seconds']!=14400 or intent['automatic_retry'] is not False
            or launch['source_commit']!=intent['source_commit']):
        raise ValueError('successful exact bounded CPU container required')


def check_records(records,rows,checks,response_ids):
    expected=[(r['id'],r['input']['decisions'][0]['id']) for r in rows]
    if (len(rows)!=6 or [(r['example_id'],r['decision_id']) for r in records]!=expected
            or len(set(expected))!=6):
        raise ValueError('all six ordered first decisions required')
    for record,row,key in zip(records,rows,expected,strict=True):
        if (record['audio_count']!=len(row['audio_sha256'])
                or record['prompt_tokens']!=checks[key]['input_tokens']
                or record['response_token_ids']!=response_ids
                or len(record['response_log_probs'])!=len(response_ids)
                or any(type(v) not in (float,int) or not math.isfinite(v) or v>0
                       for v in record['response_log_probs'])
                or type(record['seconds']) not in (float,int) or not math.isfinite(record['seconds'])
                or record['seconds']<=0 or type(record['peak_rss_kib']) is not int
                or record['peak_rss_kib']<=0):
            raise ValueError('invalid input-bound finite forward result')
    if {r['audio_count'] for r in records}!={1,2}:raise ValueError('single and double audio required')


def accept(root,launch_hash,intent_hash,tokenizer,inspection=None):
    if file_hash(root/'launch.json')!=launch_hash or file_hash(root/'intent.json')!=intent_hash:
        raise ValueError('registered launch/intent changed')
    launch=read(root/'launch.json');intent=read(root/'intent.json')
    cid=launch['container_id']
    if not isinstance(cid,str) or len(cid)!=64 or any(c not in '0123456789abcdef' for c in cid):
        raise ValueError('exact Docker ID required')
    state=inspect(cid) if inspection is None else inspection
    check_container(state,launch,intent)
    if file_hash(root/'config.json')!=intent['config_sha256'] or launch['config_sha256']!=intent['config_sha256']:
        raise ValueError('configuration changed')
    source=root/'source'
    if {str(p.relative_to(source)) for p in source.rglob('*.py')}!=set(intent['source_files']):
        raise ValueError('source roster changed')
    for name,pin in intent['source_files'].items():
        if file_hash(source/name)!=pin:raise ValueError('run source changed')
    # New verifier may live outside the original snapshot, but reused input
    # validation must have exactly the same source as the executing runner.
    import sys
    cfg=read(root/'config.json');base,rows,preflight,checks=load_inputs(cfg)
    for name,module in list(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if path is None:continue
        path=Path(path).resolve();repo=Path(__file__).resolve().parents[2]
        if (path.is_relative_to(repo/'src/multimodal_jev') or path.is_relative_to(repo/'tools/jev')) and path!=Path(__file__).resolve():
            relative=str(path.relative_to(repo))
            if intent['source_files'].get(relative)!=file_hash(path):
                raise ValueError('verifier input dependency differs from run')
    run=root/'run';identity=read(run/'identity.json');result=read(run/'result.json')
    asset=read(base['asset_audit'])
    if (asset['status']!='verified_inference_assets' or asset['model']!=base['audio_model']
            or asset['repository']!='Qwen/Qwen3-Omni-30B-A3B-Instruct'
            or asset['revision']!='26291f793822fb6be9555850f06dfe95f2d7e695'
            or asset['all_publisher_files_present'] is not True or not asset['files']):
        raise ValueError('original complete model asset audit required')
    expected_keys=[[r['id'],r['input']['decisions'][0]['id']] for r in rows]
    expected_sources={k:v for k,v in intent['source_files'].items()
                      if k.startswith(('src/multimodal_jev/','tools/jev/')) and k.count('/')==2}
    if (identity['config']!=cfg or identity['versions']!=base['versions']
            or identity['selected_keys']!=expected_keys or identity['model_files']!=asset['files']
            or identity['source_files']!=expected_sources or identity['targets_read'] is not False
            or identity['response_text']!='false'
            or identity['model_arguments']!=dict(dtype='bfloat16',device_map='cpu',attn_implementation='sdpa')
            or result['identity_sha256']!=file_hash(run/'identity.json')
            or result['loading_info_sha256']!=file_hash(run/'loading-info.json')
            or result['status']!='passed' or result['model_loaded'] is not True
            or result['weight_updates']!=0 or result['targets_read'] is not False
            or result['parameters']!=31719205488 or read(run/'terminal.json')['exit_code']!=0):
        raise ValueError('pretrained model/result identity differs')
    pins=[dict(example_id=k[0],decision_id=k[1],**{n:checks[tuple(k)][n]
          for n in ('prompt_sha256','input_ids_sha256')}) for k in expected_keys]
    if identity['prompt_evidence']!=pins:raise ValueError('registered prompts differ')
    loading=read(run/'loading-info.json')
    if (any(loading.get(k) for k in ('missing_keys','mismatched_keys','error_msgs'))
            or any(not k.startswith(('talker.','code2wav.')) for k in loading.get('unexpected_keys',[]))):
        raise ValueError('incomplete pretrained Thinker load')
    for name,pin in preflight['processor_assets'].items():
        if file_hash(Path(base['audio_model'])/name)!=pin:raise ValueError('processor assets changed')
    response=tokenizer.encode('false',add_special_tokens=False)+[tokenizer.eos_token_id]
    check_records(result['responses'],rows,checks,response)
    if (type(result['elapsed_seconds']) not in (int,float) or not math.isfinite(result['elapsed_seconds'])
            or not 0<result['elapsed_seconds']<=14400 or type(result['peak_rss_kib']) is not int
            or result['peak_rss_kib']<max(r['peak_rss_kib'] for r in result['responses'])):
        raise ValueError('complete bounded resource accounting required')
    if sorted(p.name for p in run.glob('forward-*.json'))!=[f'forward-{i:02d}.json' for i in range(6)]:
        raise ValueError('exact six forward artifacts required')
    for i,record in enumerate(result['responses']):
        if read(run/f'forward-{i:02d}.json')!=record:raise ValueError('forward artifact differs')
    return dict(schema='omni-pretrained-cpu-forward-acceptance-v1',container_id=cid,
        image=state['Image'],source_commit=launch['source_commit'],result_sha256=file_hash(run/'result.json'),
        launch_sha256=launch_hash,intent_sha256=intent_hash,parameters=result['parameters'],
        forwards=6,audio_counts=[r['audio_count'] for r in result['responses']],
        peak_rss_kib=result['peak_rss_kib'],elapsed_seconds=result['elapsed_seconds'],weight_updates=0,
        verifier_sha256=file_hash(__file__),scope='Frozen full pretrained CPU Thinker finite forward on '
            'registered TRAIN audio and fixed synthetic response tokens. No generation quality, vLLM parity, '
            'verl actor, NPU or training acceptance. Model file hashes are recorded by the pinned runner.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    for key in ('launch-hash','intent-hash'):p.add_argument('--'+key,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--inspection',type=Path);p.add_argument('--inspection-hash');a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve prior acceptance')
    inspection=None
    if a.inspection is not None or a.inspection_hash is not None:
        if a.inspection is None or not a.inspection_hash or file_hash(a.inspection)!=a.inspection_hash:
            raise ValueError('exact coordinator-captured Docker inspection required')
        inspection=read(a.inspection)
    from transformers import AutoTokenizer
    cfg=read(a.root/'config.json');base=read(cfg['baseline_config'])
    tokenizer=AutoTokenizer.from_pretrained(base['audio_model'],local_files_only=True)
    result=accept(a.root,a.launch_hash,a.intent_hash,tokenizer,inspection)
    if a.inspection is not None:result['inspection_sha256']=a.inspection_hash
    atomic_json(a.output,result)
    print(json.dumps(result,indent=2))
