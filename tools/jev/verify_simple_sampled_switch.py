#!/usr/bin/env python3
"""Independent verification of one real resident engine's old/new/old results."""
import argparse,json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.span_parity import compare_span_policies


def compare_behavior(behavior,accepted):
    for k in ('base_id','adapter_id','slot','backend','worker'):
        if behavior[k]!=accepted[k]:raise ValueError('fresh resident behavior version differs')
    if behavior['passed'] is not True or behavior.get('reference_compared') is not False:
        raise ValueError('explicit reference-free execution required')
    strip=lambda rows:[{k:v for k,v in r.items() if k!='parity'} for r in rows]
    if (len(behavior['records'])!=256 or len(accepted['records'])!=256
            or strip(behavior['records'])!=strip(accepted['records'])
            or any(r.get('parity') is not None for r in behavior['records'])):
        raise ValueError('fresh behavior must match accepted full256 exactly')
    return dict(passed=True,requests=256,exact_resident_behavior=True,
        scope='Reference-free collection equals separately verified resident output; actor admission remains required.')


def compare_results(results,references,*,base_id,adapter_ids):
    if len(results)!=3 or len(references)!=2 or len(adapter_ids)!=2 or adapter_ids[0]==adapter_ids[1]:
        raise ValueError('distinct old/new/old sequence required')
    checks=[]
    for phase,index in enumerate((0,1,0)):
        result=results[phase];reference=references[index];slot=index+1
        if (result['passed'] is not True or result['backend']!='vllm' or result['base_id']!=base_id
                or result['adapter_id']!=adapter_ids[index] or result['slot']!=slot
                or reference['backend']!='hf' or reference['base_id']!=base_id
                or reference['adapter_id']!=adapter_ids[index]):
            raise ValueError('executed backend/version/slot differs')
        worker=result['worker']
        if (len(worker)!=1 or worker[0]['registered_adapter_ids']!=[slot]
                or worker[0]['lora_dtype']!='torch.bfloat16' or worker[0]['head_projection_dtype']!='torch.bfloat16'
                or not worker[0]['actual_lora_buffer_dtypes']
                or any(v!=['torch.bfloat16'] for v in worker[0]['actual_lora_buffer_dtypes'].values())):
            raise ValueError('actual worker adapter and BF16 buffers required')
        refs={r['id']:r for r in reference['records']};rows={r['id']:r for r in result['records']}
        if len(reference['records'])!=256 or len(result['records'])!=256 or len(refs)!=256 or refs.keys()!=rows.keys():
            raise ValueError('exact complete256 identity coverage required')
        for uid,row in rows.items():
            ref=refs[uid]
            if row['request_id']!=ref['request_id'] or row['request_identity']!=ref['request_identity']:
                raise ValueError('switch input/policy binding differs')
            check=compare_span_policies(row['candidate_logprobs'],ref['candidate_logprobs'])
            if not check['passed']:raise ValueError('switch full-candidate parity failed')
            checks.append(check)
    if results[0]['records']!=results[2]['records']:
        raise ValueError('switch back did not reproduce original output exactly')
    return dict(passed=True,requests_per_phase=256,phases=3,exact_switch_back=True,
        maximum_logprob_error=max(c['maximum_logprob_error'] for c in checks),
        argmax_changed_spans=sum(len(c['argmax_changed_spans']) for c in checks))


def verify(root,observed_path):
    from verify_slurp_simple_parity import verify as verify_gate
    root=Path(root);files={}
    def read(p):
        p=Path(p);files[str(p)]=file_hash(p);return json.loads(p.read_text())
    config=read(root/'config.json');launch=read(root/'launch.json');observed=read(observed_path)
    if (launch['config_sha256']!=file_hash(root/'config.json') or observed['container_id']!=launch['container_id']
            or observed['image']!=launch['image'] or observed['state']['Status']!='exited'
            or observed['state']['ExitCode']!=0 or read(root/'terminal.json')['exit_code']!=0
            or read(root/'run/terminal.json')['exit_code']!=0):
        raise ValueError('exact completed switch process required')
    for p,h in config['pins'].items():
        if file_hash(p)!=h:raise ValueError('switch dependency changed: '+p)
        files[p]=h
    gates=[Path(config[k]) for k in ('old_gate','updated_gate')]
    accepted=[verify_gate(g) for g in gates]
    if (accepted[0]['updates']!=2 or accepted[1]['updates']!=1
            or accepted[1]['sampled_resident_parity'] is None):raise ValueError('accepted sampled policy pair required')
    for receipt in accepted:files.update(receipt['files'])
    configs=[read(g/'config.json') for g in gates]
    for k in ('model','base_id','runtime_profile','max_model_len','inputs','population'):
        if configs[0][k]!=configs[1][k]:raise ValueError('same switch runtime/input required')
    state=read(configs[1]['checkpoint_evidence'])
    if state['reference_policy_id']!=configs[0]['base_id']+':'+configs[0]['adapter_id']:
        raise ValueError('fixed reference lineage differs')
    report=compare_results([read(root/'run'/f'policy-{i}.json') for i in range(3)],
        [read(g/'run/hf/result.json') for g in gates],base_id=configs[0]['base_id'],
        adapter_ids=[c['adapter_id'] for c in configs])
    behavior_check=compare_behavior(read(root/'run/behavior.json'),read(root/'run/policy-1.json'))
    return dict(**report,files=files,container_id=launch['container_id'],
        reference_free_behavior=behavior_check,
        scope='Independently recomputed resident vLLM full256 old/new/old evidence; not continuous RL updates or task efficacy.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--observed',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(a.output,verify(a.root,a.observed))
