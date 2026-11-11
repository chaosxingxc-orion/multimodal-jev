#!/usr/bin/env python3
"""Audit every version and actual optimizer cursor in a completed trajectory."""
import argparse,json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.span_transfer_runtime import branch_protocol
from multimodal_jev.span_training_chain import checkpoint_receipt
from verify_span_rule_transfer import verify as verify_protocol
from verify_span_rule_baseline import verify as verify_hf
from verify_span_rule_runtime import verify as verify_runtime
from verify_span_transfer_step import verify as verify_step


def verify(root):
    read=lambda p:json.loads(p.read_text());c=read(root/'config.json');launch=read(root/'launch.json');run=root/'run'
    from multimodal_jev.span_runtime_topology import runtime_topology
    devices=c.get('physical_devices',[5])
    if (runtime_topology({**c,'physical_devices':devices})['size']!=1
            or launch['launch_exit_code']!=0 or launch['physical_devices']!=devices
            or read(run/'terminal.json')['exit_code']!=0
            or launch['config_sha256']!=file_hash(root/'config.json')
            or launch['source_manifest_sha256']!=file_hash(root/'source-manifest.json')
            or launch['protocol_sha256']!=file_hash(Path(c['transfer_protocol']))
            or launch['controller_sha256']!=file_hash(root/'source/tools/jev/run_span_transfer_chain.py')):
        raise ValueError('chain launch or terminal differs')
    for path,digest in c['pins'].items():
        if file_hash(path)!=digest:raise ValueError('chain pin changed')
    source=read(root/'source-manifest.json')
    for name,digest in source['files'].items():
        if file_hash(root/'source'/name)!=digest:raise ValueError('chain snapshot changed')
    protocol=branch_protocol(c);verify_protocol(Path(c['transfer_protocol']))
    count=c['num_updates'];start=c.get('start_index',0)
    initial=protocol['initializer']['adapter_id'] if start==0 else checkpoint_receipt(Path(c['initial_optimizer_checkpoint']))[0]['adapter_id']
    versions=[];previous_id=initial
    for version in range(start,start+count+1):
        stage=run/f'version-{version:04d}';hf=verify_hf(stage);runtime=verify_runtime(stage/'runtime')
        if not runtime['passed'] or runtime['adapter_id']!=hf['adapter']:
            raise ValueError('runtime gate failed')
        cfg=read(stage/'config.json')
        for key in ('transfer_protocol','seed','method','arm','auxiliary_weight'):
            if cfg[key]!=c[key]:raise ValueError('runtime branch differs')
        if cfg['version_index']!=version:raise ValueError('runtime version order differs')
        entry=dict(version=version,adapter_id=runtime['adapter_id'],requests=runtime['requests'],
            maximum_logprob_error=runtime['maximum_logprob_error'])
        if version==start:
            if hf['adapter']!=initial:raise ValueError('initializer/resume policy differs')
        else:
            update=run/f'update-{version:04d}';uc=read(update/'config.json')
            if uc['batch_index']!=version-1 or uc['adapter_id']!=previous_id:
                raise ValueError('nonconsecutive update')
            if uc['runtime_admission']!=str(run/f'version-{version-1:04d}'/'runtime'):
                raise ValueError('update consumed different runtime')
            if version>start+1 and uc['initial_optimizer_checkpoint']!=str(run/f'update-{version-1:04d}'/'run/policy-1'):
                raise ValueError('optimizer parent differs')
            if version==start+1 and start and uc['initial_optimizer_checkpoint']!=c['initial_optimizer_checkpoint']:
                raise ValueError('external resume parent differs')
            step=verify_step(update,stage)
            if step['adapter_id']!=hf['adapter']:raise ValueError('exported adapter differs')
            entry['step']=step
        previous_id=hf['adapter'];versions.append(entry)
    last=read(run/'last-accepted.json');ready=read(run/'READY.json')
    if (ready['status']!='passed' or ready['latest']!=last or last['completed_updates']!=start+count
            or last['adapter_id']!=previous_id
            or last['step_audit_sha256']!=file_hash(run/f'update-{start+count:04d}'/'step-audit.json')):
        raise ValueError('terminal acceptance pointer differs')
    return dict(passed=True,root=str(root),versions=versions,initial_completed_updates=start,
        completed_updates=start+count,optimizer_updates=count,protocol_sha256=file_hash(Path(c['transfer_protocol'])),
        scope='Actual bounded optimizer lineage, exact independent reload and every-version HF/vLLM coverage; no academic metric or cross-task gain assertion')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=verify(a.root);atomic_json(a.output,r);print(json.dumps(r,indent=2))
