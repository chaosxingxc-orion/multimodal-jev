#!/usr/bin/env python3
"""Bounded transfer trajectory with a fresh process and full gate per version."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.policy_identity import checkpoint_identity
from multimodal_jev.span_transfer_runtime import branch_protocol,version_rows,POPULATION
from multimodal_jev.span_rule_population import select_rule_rows,LEARNING_PROBE
from multimodal_jev.span_training_chain import checkpoint_receipt
from verify_span_rule_transfer import verify as verify_protocol
from verify_span_rule_runtime import verify as verify_runtime
from verify_span_transfer_step import verify as verify_step


def run(config,output):
    for path,digest in config['pins'].items():
        if file_hash(path)!=digest:raise ValueError('chain input changed')
    p=branch_protocol(config);atomic_json(output/'protocol-audit.json',verify_protocol(Path(config['transfer_protocol'])))
    count=config['num_updates'];start=config.get('start_index',0)
    if type(count) is not int or count<1 or type(start) is not int or not 0<=start<start+count<=p['updates_per_branch']:
        raise ValueError('bounded trajectory within protocol required')
    tools=Path(__file__).parent;source=tools.parents[1]
    read=lambda path:json.loads(path.read_text())
    common={**config,'population':POPULATION,'physical_devices':[5]}
    common['pins']=dict(config['pins']);common['base_id']=checkpoint_identity(config['model'])[0]
    if common['base_id']!=p['initializer']['base_id']:raise ValueError('wrong base')
    if start==0:
        adapter=Path(p['initializer']['root'])/'run/adapter';checkpoint=None
        if config.get('initial_optimizer_checkpoint'):raise ValueError('fresh trajectory required')
    else:
        checkpoint=Path(config['initial_optimizer_checkpoint']);receipt,pins=checkpoint_receipt(checkpoint,config['pins'])
        from multimodal_jev.span_transfer_runtime import validate_parent
        validate_parent({**config,'batch_index':start,'adapter_id':receipt['adapter_id']},receipt)
        adapter=checkpoint/'adapter'
    manifest=read(source/'../source-manifest.json')
    launch=read(source/'../launch.json')
    def progress(stage,index):atomic_json(output/'progress.json',dict(stage=stage,version=index,updated_unix=time.time()))
    def child(script,c,dest):
        dest.mkdir(exist_ok=False);atomic_json(dest/'config.json',c)
        with (dest/'process.log').open('x') as log:
            subprocess.run([sys.executable,str(tools/script),'--config',str(dest/'config.json'),'--output',str(dest/'run')],stdout=log,stderr=subprocess.STDOUT,check=True)
        if read(dest/'run/terminal.json')['exit_code']!=0:raise ValueError('child failed')
    def gate(version,adapter):
        progress('version_hf',version);stage=output/f'version-{version:04d}';c={**common,'adapter':str(adapter),'adapter_id':checkpoint_identity(adapter)[0],'version_index':version,'pins':dict(common['pins'])}
        for f in adapter.iterdir():
            if f.is_file():c['pins'][str(f)]=file_hash(f)
        ids=[r['id'] for r in select_rule_rows(Path(c['release']),LEARNING_PROBE)]
        for i in (version-1,version):
            if 0<=i<p['updates_per_branch']:
                ids.extend(k for k in p['schedules'][str(c['seed'])][i]['arms'][c['arm']] if k not in ids)
        c['request_ids']=ids;version_rows(c)
        child('infer_span_rule_baseline.py',c,stage)
        atomic_json(stage/'source-manifest.json',manifest)
        receipt=dict(root=str(stage),image=launch['image'],physical_devices=[5],execution='stage in pinned transfer controller',
            config_sha256=file_hash(stage/'config.json'),source_manifest_sha256=file_hash(stage/'source-manifest.json'),controller_sha256=file_hash(Path(__file__)))
        atomic_json(stage/'launch.json',receipt)
        runtime=stage/'runtime';rc={**c,'hf_run':str(stage),'pins':dict(c['pins'])}
        for name in ('config.json','source-manifest.json','launch.json','run/result.json','run/records.jsonl','run/terminal.json'):
            rc['pins'][str(stage/name)]=file_hash(stage/name)
        progress('version_vllm',version);child('check_span_rule_runtime.py',rc,runtime)
        atomic_json(runtime/'source-manifest.json',manifest)
        atomic_json(runtime/'launch.json',{**receipt,'root':str(runtime),'config_sha256':file_hash(runtime/'config.json'),'source_manifest_sha256':file_hash(runtime/'source-manifest.json')})
        verified=verify_runtime(runtime)
        if not verified['passed']:raise ValueError('full runtime version failed')
        atomic_json(runtime/'verified.json',verified)
        return stage,runtime
    # Recompute resumed version's gate; never trust only a READY marker.
    stage,runtime=gate(start,adapter)
    for i in range(start,start+count):
        progress('training',i)
        c={**common,'adapter':str(adapter),'adapter_id':checkpoint_identity(adapter)[0],
            'batch_index':i,'num_updates':1,'optimizer_start':'resume' if i else 'fresh',
            'runtime_admission':str(runtime),'request_ids':p['schedules'][str(common['seed'])][i]['arms'][common['arm']],
            'pins':dict(common['pins'])}
        for f in adapter.iterdir():
            if f.is_file():c['pins'][str(f)]=file_hash(f)
        for f in runtime.rglob('*'):
            if f.is_file():c['pins'][str(f)]=file_hash(f)
        if checkpoint:
            c['initial_optimizer_checkpoint']=str(checkpoint);c['pins'].update(checkpoint_receipt(checkpoint)[1])
        update=output/f'update-{i+1:04d}';child('train_span_rule_step.py',c,update)
        audit=verify_step(update);atomic_json(update/'step-audit.json',audit)
        checkpoint=update/'run/policy-1';adapter=checkpoint/'adapter'
        stage,runtime=gate(i+1,adapter)
        audit=verify_step(update,stage);atomic_json(update/'step-audit.json',audit)
        accepted=dict(completed_updates=i+1,checkpoint=str(checkpoint),adapter_id=audit['adapter_id'],runtime=str(runtime),
            protocol_sha256=file_hash(config['transfer_protocol']),step_audit_sha256=file_hash(update/'step-audit.json'))
        atomic_json(stage/'READY.json',accepted);atomic_json(output/'last-accepted.json',accepted)
    atomic_json(output/'READY.json',dict(status='passed',latest=accepted,scope='Bounded transfer training trajectory and per-version gates; task metrics/generalization assessed separately'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        run(json.loads(a.config.read_text()),a.output)
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
