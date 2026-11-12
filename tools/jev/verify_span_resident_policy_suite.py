#!/usr/bin/env python3
"""Host verification of a completed resident suite; no torch import required."""
import argparse
import math
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.intent_gate_identity import verify_population
from multimodal_jev.policy_identity import checkpoint_identity
from multimodal_jev.slurp_intent_pilot import load_pilot
from queue_span_dp4_resume import check_pins
from queue_span_native_actor_npu import read,completed


def verify_policy(config,artifact,expected,hf,vl,slot):
    """Verify task identity independently, then every joint-action ratio bound."""
    pid=config['base_id']+':'+config['adapter_id']
    for backend,report in (('hf',hf),('vllm',vl)):
        if (report.get('passed') is not True or report['backend']!=backend
                or report['base_id']!=config['base_id'] or report['adapter_id']!=config['adapter_id']):
            raise ValueError('same-policy complete backend required')
        verify_population(artifact,expected,report['records'],pid)
    worker=vl['worker']
    if (vl['slot']!=slot or len(worker)!=1 or worker[0]['registered_adapter_ids']!=[slot]
            or worker[0]['lora_dtype']!='torch.bfloat16' or worker[0]['head_projection_dtype']!='torch.bfloat16'
            or not worker[0]['actual_lora_buffer_dtypes']
            or any(v!=['torch.bfloat16'] for v in worker[0]['actual_lora_buffer_dtypes'].values())):
        raise ValueError('actual BF16 resident adapter required')
    refs={r['id']:r for r in hf['records']};worst=0.;low=1.;high=1.
    for row in vl['records']:
        ref=refs[row['id']]
        if row['request_id']!=ref['request_id'] or row['request_identity']!=ref['request_identity']:
            raise ValueError('exact backend request identity required')
        lower=upper=0.
        for a,b in zip(ref['candidate_logprobs'],row['candidate_logprobs'],strict=True):
            delta=[x-y for x,y in zip(a,b,strict=True)]
            worst=max(worst,max(map(abs,delta)));lower+=min(delta);upper+=max(delta)
        if worst>.02 or lower<math.log(.98) or upper>math.log(1.02):
            raise ValueError('full candidate/joint parity failed')
        low=min(low,math.exp(lower));high=max(high,math.exp(upper))
    return dict(passed=True,requests=len(expected),policy_id=pid,slot=slot,
                maximum_logprob_error=worst,joint_ratio_range=[low,high])


def verify(root,pilot,manifest,image):
    root=Path(root);receipt=read(root/'launch.json');plan=read(root/'prepared.json')
    if (receipt['root']!=str(root) or receipt['image']!=image or plan['image']!=image
            or receipt['physical_devices']!=[4] or plan['physical_devices']!=[4]
            or receipt['argv']!=plan['create_argv'] or receipt['launch_exit_code']!=0
            or not completed(receipt['container_id'],image)):
        raise ValueError('exact successful resident container required')
    for name,key in (('config.json','config_sha256'),('source-manifest.json','source_manifest_sha256')):
        if file_hash(root/name)!=receipt[key] or receipt[key]!=plan[key]:raise ValueError('launched source/config changed')
    cfg=read(root/'config.json');check_pins(cfg['pins'])
    check_pins({str(root/'source'/p):h for p,h in read(root/'source-manifest.json').items()})
    if read(root/'run/terminal.json')['exit_code']!=0:raise ValueError('resident suite failed')
    summary=read(root/'run/result.json')
    if (cfg['schema']!='jev-resident-policy-suite-v1' or summary.get('passed') is not True
            or len(cfg['policies'])<2 or [r['key'] for r in summary['policies']]!=[p['key'] for p in cfg['policies']]):
        raise ValueError('complete registered policy sequence required')
    check_pins(summary['files']);expected={}
    for arm in ('D','Q'):
        items,_=load_pilot(pilot,manifest,arm)
        expected.update({i['id']+'::'+arm:i['request'].model_input() for i in items})
    if len(expected)!=178:raise ValueError('full reviewed89-source population required')
    pins=dict(summary['files']);audits=[];slots={};seen=set();common=None
    for entry,saved in zip(cfg['policies'],summary['policies'],strict=True):
        key=entry['key']
        if (not isinstance(key,str) or not key or key in seen
                or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in key)):
            raise ValueError('distinct registered safe policy key required')
        seen.add(key);source=Path(entry['config']);sub=root/'run'/key;c=read(sub/'config.json')
        if cfg['pins'].get(str(source))!=file_hash(source) or c!=read(source):
            raise ValueError('policy configuration differs from registered reference')
        check_pins(c['pins'])
        if (c['physical_devices']!=[4] or c['max_model_len']!=4096
                or c['population']!=dict(recordings=89,requests=178)
                or checkpoint_identity(c['model'])[0]!=c['base_id']
                or checkpoint_identity(c['adapter'])[0]!=c['adapter_id']
                or c['pins'].get(c['inputs'])!=file_hash(c['inputs'])):
            raise ValueError('policy/runtime/population identity differs')
        current={k:c[k] for k in ('model','base_id','runtime_profile','max_model_len','inputs','population')}
        if common is None:common=current
        if current!=common:raise ValueError('resident runtime or population changed')
        if c['adapter_id'] not in slots:slots[c['adapter_id']]=len(slots)+1
        for name in ('terminal.json','hf/terminal.json','vllm/terminal.json'):
            if read(sub/'run'/name)['exit_code']!=0:raise ValueError('policy phase failed')
        ready=read(sub/'run/READY.json');hf=read(sub/'run/hf/result.json');vl=read(sub/'run/vllm/result.json')
        if (ready.get('passed') is not True or ready['requests']!=178
                or ready['base_id']!=c['base_id'] or ready['adapter_id']!=c['adapter_id']
                or ready['result_sha256']!=file_hash(sub/'run/vllm/result.json')
                or read(sub/'run/vllm/worker.json')!=vl['worker']):
            raise ValueError('READY or worker differs')
        for profile in (sub/'run/hf/runtime-profile.json',sub/'run/vllm/runtime-profile.json',root/'run/runtime-profile.json'):
            if read(profile)['profile']!=c['runtime_profile']:raise ValueError('actual runtime profile differs')
            pins[str(profile)]=file_hash(profile)
        audit=verify_policy(c,read(c['inputs']),expected,hf,vl,slots[c['adapter_id']])
        if saved!={k:v for k,v in dict(key=key,adapter_id=c['adapter_id'],**audit).items()
                   if k in ('key','adapter_id','slot','requests','maximum_logprob_error')}:
            raise ValueError('summary differs from independent recomputation')
        required=[sub/'config.json',*[sub/'run'/n for n in ('READY.json','terminal.json','hf/result.json','hf/terminal.json','vllm/result.json','vllm/terminal.json')]]
        if any(summary['files'].get(str(p))!=file_hash(p) for p in required):
            raise ValueError('complete per-policy output pins required')
        pins[str(sub/'run/vllm/worker.json')]=file_hash(sub/'run/vllm/worker.json')
        audits.append(dict(key=key,gate_root=str(sub),**audit))
    for p in (root/'launch.json',root/'prepared.json',root/'config.json',root/'source-manifest.json',root/'run/terminal.json',root/'run/result.json',Path(__file__)):
        pins[str(p)]=file_hash(p)
    return dict(passed=True,root=str(root),container_id=receipt['container_id'],image=image,
        policies=audits,files=pins,pilot=str(pilot),manifest_sha256=manifest,
        scope='Actual completed multi-policy inference only. No optimizer state acceptance or efficacy claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','pilot','manifest','image','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError('immutable acceptance exists')
    atomic_json(out,verify(a.root,a.pilot,a.manifest,a.image))
