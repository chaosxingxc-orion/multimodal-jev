"""Run one registered finite diagnostic after an exact audited predecessor.

No periodic polling, retries, model calls, policy updates or training restart.
Dispatch retains the diagnostic's shared resource lease and idle-device checks.
"""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.finalize_context_infonce import wait_exact

MODULES={'language_stage':'tools.reports.context_language_stage_operations',
         'attention_stage':'tools.reports.context_attention_stage_operations',
         'attention_kernel':'tools.reports.context_attention_kernel_operations',
         'bounded_attention':'tools.reports.context_bounded_attention_operations'}
RESULTS={'language_stage':'language_tensors_independently_compared_not_admission',
         'attention_stage':'attention_boundaries_independently_compared_not_admission',
         'attention_kernel':'attention_kernel_independently_compared_not_admission',
         'bounded_attention':'bounded_attention_full_model_compared_not_admission'}


def read(path):return json.loads(Path(path).read_text())


def validate(c):
    if c.get('diagnostic_dispatch_allowed') is not True or c['diagnostic'] not in MODULES:
        raise ValueError('explicit registered finite diagnostic required')
    required={c['predecessor_driver'],c['predecessor_config'],c['operations_config'],c['operations_local']}
    if not required<=set(c['pins']):raise ValueError('all predecessor and operation inputs must be pinned')
    for name,digest in c['pins'].items():
        if file_hash(name)!=digest:raise ValueError('diagnostic dependency changed: '+name)
    prior=read(c['predecessor_config'])
    if c['predecessor_terminal']!=str(Path(prior['finalization_root'])/'terminal.json'):
        raise ValueError('exact InfoNCE finalization predecessor required')
    ops,provenance=load_config(c['operations_config'],c['operations_local'])
    training,_=load_config(prior['operations_config'],prior['operations_local'])
    if any(ops[k]!=training[k] for k in ('host','lease')):
        raise ValueError('same host and exclusive four-device lease required')
    module=importlib.import_module(MODULES[c['diagnostic']]);module.validate(ops)
    base=Path(ops['operations_root'])
    if any((base/(ops['tag']+s)).exists() for s in ('-attempt.json','-launch.json')):
        raise ValueError('already attempted diagnostic; reconcile original handles')
    check_paths({'diagnostic_successor':c['output']})
    return module,ops,provenance


def run(c,provenance):
    validate(c);root=Path(c['output']);root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'config.json',c);commit_json(root/'provenance.json',provenance)
    try:
        record=read(c['predecessor_driver']);commit_json(root/'waiting.json',record)
        wait_exact(record,c['predecessor_terminal'])
        state=read(c['predecessor_terminal'])
        if (state.get('exit_code')!=0 or state.get('status')!='formal_infonce_audited_curve_bound'
                or state.get('points')!=11 or state.get('evaluation_complete') is not False):
            raise ValueError('successful full InfoNCE audit and bound curve required')
        module,ops,p=validate(c)
        commit_json(root/'dispatch.json',dict(operations=ops,predecessor_terminal_sha256=file_hash(c['predecessor_terminal'])))
        result=module.launch(ops,p)
        if result.get('status')!=RESULTS[c['diagnostic']]:
            raise ValueError('independent diagnostic tensor comparison required')
        commit_json(root/'result.json',result)
        commit_json(root/'terminal.json',dict(exit_code=0,status='finite_diagnostic_audited',
            training_dispatched=False,restart=False))
        return result
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False,preserve_handles=True));raise


def detach(template,local):
    c,_=load_config(template,local);validate(c)
    root=Path(c['output']).with_name(Path(c['output']).name+'-driver')
    check_paths({'diagnostic_driver':root});root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-B','-m','tools.reports.run_context_diagnostic_successor',
        '--config',str(Path(template).resolve()),'--local',str(Path(local).resolve()),'--action','run']
    with (root/'driver.log').open('x') as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=child.pid,argv=argv,started_unix=time.time(),
        process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19])
    commit_json(root/'started.json',record);return record


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['validate','run','detach'],default='validate');a=p.parse_args()
    c,provenance=load_config(a.config,a.local)
    if a.action=='detach':result=detach(a.config,a.local)
    elif a.action=='run':result=run(c,provenance)
    else:validate(c);result=dict(status='validated_only_no_dispatch')
    print(json.dumps(result,indent=2))
