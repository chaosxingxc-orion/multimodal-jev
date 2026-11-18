#!/usr/bin/env python3
"""Independently verify fresh-process RL recovery state and all256 probabilities."""
import argparse,json,time
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash


def run(config,output,*,expected_mode='restore_updated_rl_only'):
    import torch
    from multimodal_jev.intent_resident_replay import compare_state
    from multimodal_jev.span_parity import compare_span_policies
    from verify_slurp_simple_parity import verify
    if list(Path('/dev').glob('davinci[0-9]*')):raise ValueError('CPU-only independent audit required')
    if expected_mode not in ('restore_updated_rl_only','restore_updated_rl_then_second_update',
                             'restore_updated_rl_then_resident_loop'):
        raise ValueError('registered recovery mode required')
    files={}
    def read(p):
        p=Path(p);files[str(p)]=file_hash(p);return json.loads(p.read_text())
    for p,h in config['pins'].items():
        if file_hash(p)!=h:raise ValueError('audit dependency changed: '+p)
        files[p]=h
    root=Path(config['training_root']);training=read(root/'config.json');launch=read(root/'launch.json')
    observed=read(config['training_observed'])
    if (observed['container_id']!=launch['container_id'] or observed['image']!=launch['image']
            or observed['state']['Status']!='exited' or observed['state']['ExitCode']!=0
            or launch['config_sha256']!=file_hash(root/'config.json') or read(root/'terminal.json')['exit_code']!=0
            or training['sampled_probe']['fixture_initialization']!=expected_mode):
        raise ValueError('exact successful independent recovery process required')
    for p,h in training['pins'].items():
        if file_hash(p)!=h:raise ValueError('recovery dependency changed: '+p)
        files[p]=h
    gate_root=Path(training['sampled_probe']['behavior_gate']);gate=verify(gate_root);files.update(gate['files'])
    if gate['updates']!=1 or gate['requests']!=256 or gate['sampled_resident_parity'] is None:
        raise ValueError('accepted sampled RL policy required')
    gate_config=read(gate_root/'config.json');artifact=read(gate_config['inputs'])
    source_root=Path(gate['training_root']);source_config=read(source_root/'config.json')
    from multimodal_jev.slurp_instruction_resume import FIELDS
    if any(training[k]!=source_config[k] for k in FIELDS):raise ValueError('recovery factors differ')
    if training['sampled_probe']['algorithm']!=source_config['sampled_probe']['algorithm']:
        raise ValueError('algorithm changed')
    references={b:{r['id']:r for r in read(gate_root/'run'/b/'result.json')['records']} for b in ('hf','vllm')}
    checks=[];state_checks=[]
    for rank in range(4):
        out=root/'run'/f'rank-{rank}'
        report=read(out/'result.json');probabilities=read(out/'restored-policy.json')
        if (read(out/'terminal.json')['exit_code']!=0 or report['passed'] is not True
                or report['rank']!=rank or report['updates']!=1 or report['adapter_id']!=gate['adapter_id']
                or report['base_id']!=training['base_id'] or len(report['execution_dtypes'])!=112
                or any(v!=['torch.bfloat16','torch.bfloat16'] for v in report['execution_dtypes'].values())):
            raise ValueError('complete BF16 recovered rank required')
        source=source_root/'run'/f'rl-rank-{rank}/checkpoint-0001.pt';saved=out/'checkpoint-0001.pt'
        if report['checkpoint_source']!=str(source):raise ValueError('recovery checkpoint source differs')
        for p in (source,saved):files[str(p)]=file_hash(p)
        states=[torch.load(p,map_location='cpu',weights_only=True) for p in (source,saved)]
        for state in states:
            if (state['adapter_id']!=gate['adapter_id'] or state['algorithm']!=training['sampled_probe']['algorithm']
                    or state['reference_policy_id']!=source_config['base_id']+':'+read(Path(source_config['sampled_probe']['behavior_gate'])/'config.json')['adapter_id']):
                raise ValueError('recovered state provenance differs')
        state_checks.append(compare_state(*states,rank=rank,updates=1))
        rows=probabilities['records'];expected=[r['id'] for r in artifact['records'][rank::4]]
        if len(rows)!=64 or [r['id'] for r in rows]!=expected:raise ValueError('complete exact per-rank input order required')
        for row in rows:
            for backend in ('hf','vllm'):
                result=compare_span_policies(row['candidate_logprobs'],references[backend][row['id']]['candidate_logprobs'])
                if not result['passed']:raise ValueError('fresh recovery/backend probabilities differ')
                checks.append(result)
    return dict(passed=True,updates=1,ranks=4,requests=256,adapter_id=gate['adapter_id'],
        exact_master_adam=True,state_checks=state_checks,full_candidate_parity=True,
        maximum_logprob_error=max(c['maximum_logprob_error'] for c in checks),
        argmax_changed_spans=sum(len(c['argmax_changed_spans']) for c in checks),files=files,
        scope='Independently recomputed fresh-process recovered FP32 master/Adam and all256 FSDP/HF/vLLM probabilities; no next optimizer update or efficacy claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    try:
        atomic_json(a.output/'state-verified.json',run(json.loads(a.config.read_text()),a.output))
        atomic_json(a.output/'terminal.json',dict(exit_code=0,finished_unix=time.time()))
    except BaseException as exc:
        atomic_json(a.output/'terminal.json',dict(exit_code=1,error=repr(exc),finished_unix=time.time()));raise
