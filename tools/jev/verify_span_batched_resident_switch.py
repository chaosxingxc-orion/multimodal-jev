#!/usr/bin/env python3
"""Independent candidate resident switch evidence; execution receipts required."""
import argparse,json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from verify_simple_sampled_switch import compare_behavior,compare_results


def verify(root,observed_path,launch_path):
    from verify_slurp_batched_parity import verify as verify_gate
    root=Path(root);files={}
    def read(p):
        p=Path(p);files[str(p)]=file_hash(p);return json.loads(p.read_text())
    config=read(root/'config.json');launch=read(launch_path);observed=read(observed_path)
    if (launch['config_sha256']!=file_hash(root/'config.json') or observed['container_id']!=launch['container_id']
            or observed['image']!=launch['image'] or observed['state']['Status']!='exited'
            or observed['state']['ExitCode']!=0 or read(root/'terminal.json')['exit_code']!=0
            or read(root/'run/terminal.json')['exit_code']!=0):
        raise ValueError('exact completed switch process required')
    for p,h in config['pins'].items():
        if file_hash(p)!=h:raise ValueError('switch dependency changed: '+p)
        files[p]=h
    audits=[Path(config[k]) for k in ('old_audit_config','updated_audit_config')]
    if any(config['pins'].get(str(p))!=file_hash(p) for p in audits):
        raise ValueError('pinned independent candidate audit configs required')
    accepted=[verify_gate(p) for p in audits]
    if ([a['updates'] for a in accepted]!=[2,4]
            or any(a['requests']!=256 or a['population_mode']!='prefix' for a in accepted)):
        raise ValueError('accepted candidate step2 and step4 prefix policies required')
    for receipt in accepted:files.update(receipt['files'])
    gates=[Path(read(p)['backend_root']) for p in audits]
    configs=[read(g/'config.json') for g in gates]
    for k in ('model','base_id','runtime_profile','max_model_len','inputs','population'):
        if configs[0][k]!=configs[1][k]:raise ValueError('same switch runtime/input required')
    from multimodal_jev.span_batch_training_config import PROFILE
    if config.get('candidate_profile')!=PROFILE:
        raise ValueError('candidate execution contract required')
    runtime=read(root/'run/runtime-profile.json')
    if runtime.get('candidate_profile')!=PROFILE:
        raise ValueError('candidate installation receipt required')
    report=compare_results([read(root/'run'/f'policy-{i}.json') for i in range(3)],
        [read(g/'run/hf/result.json') for g in gates],base_id=configs[0]['base_id'],
        adapter_ids=[c['adapter_id'] for c in configs])
    behavior_check=compare_behavior(read(root/'run/behavior.json'),read(root/'run/policy-1.json'))
    return dict(**report,files=files,container_id=launch['container_id'],
        reference_free_behavior=behavior_check,
        scope='Independently recomputed candidate resident step2/step4/step2 full256 evidence; not RL admission or efficacy.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--observed',type=Path,required=True)
    p.add_argument('--launch',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(a.output,verify(a.root,a.observed,a.launch))
