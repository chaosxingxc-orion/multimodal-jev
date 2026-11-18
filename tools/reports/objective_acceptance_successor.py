"""Wait on the exact existing controller, then run one finite acceptance.

No polling, preemption, retries, model calls, or modification of the predecessor.
Failure of the predecessor stops this successor before any NPU dispatch.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.prepare_objective_training_candidate import read_bound
from tools.reports.finalize_context_infonce import wait_exact
from tools.reports import run_objective_acceptance as sequence


def validate(c):
    prior=json.loads(read_bound(c['predecessor_config']))
    driver=json.loads(read_bound(c['predecessor_driver']))
    path=Path(prior['output'])
    if (c['predecessor_terminal']!=str(path/'terminal.json')
            or c['predecessor_driver']['path']!=str(path.with_name(path.name+'-driver')/'started.json')
            or 'tools.reports.context_stage1_priority_pipeline' not in driver['argv']):
        raise ValueError('exact current four-card controller required')
    template=c['sequence_template'];local=c['sequence_local']
    read_bound(template);read_bound(local)
    config,provenance=load_config(template['path'],local['path'])
    sequence.validate(config)
    if (config['prefix']['lease']!=prior['deployment']['lease']
            or config['prefix']['host']!=prior['deployment']['host']):
        raise ValueError('same predecessor device lease and host required')
    root=Path(c['output']).resolve()
    protected=[path,Path(config['output']),Path(config['prefix']['operations_root']),
               Path(config['prefix']['package']),Path(config['recovery']['package']),
               Path(config['recovery']['operations_root'])]
    if any(root.is_relative_to(p.resolve()) or p.resolve().is_relative_to(root) for p in protected):
        raise ValueError('separate successor records required')
    for binding in c['pins']:read_bound(binding)
    return driver,config,provenance


def run(c,provenance):
    driver,config,child_provenance=validate(c);root=Path(c['output'])
    check_paths({'successor':root});root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'config.json',c);commit_json(root/'provenance.json',provenance)
    try:
        commit_json(root/'waiting.json',dict(process=driver,terminal=c['predecessor_terminal'],
            mechanism='exact_pidfd',remote_polling=False))
        wait_exact(driver,c['predecessor_terminal'])
        validate(c)
        terminal=json.loads(Path(c['predecessor_terminal']).read_text())
        if terminal.get('exit_code')!=0 or terminal.get('status')!='async_simple_pipeline_completed':
            raise ValueError('predecessor not successfully complete; no acceptance dispatch')
        # The predecessor terminal only gates resource sequencing. Every
        # actual model/reference/optimizer prerequisite is independently
        # reconstructed by the acceptance sequence itself.
        commit_json(root/'predecessor-complete.json',terminal)
        result=sequence.run(config,child_provenance)
        if result['status']!='objective_prefix_and_recovery_audited' or result['exit_code']!=0:
            raise ValueError('finite acceptance incomplete')
        commit_json(root/'result.json',result)
        commit_json(root/'terminal.json',dict(exit_code=0,status='objective_successor_completed',
            formal_training_admitted=False))
        return result
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False,preserve_handles=True))
        raise


def detach(template,local):
    c,_=load_config(template,local);validate(c)
    root=Path(c['output']).with_name(Path(c['output']).name+'-driver')
    check_paths({'driver':root});root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-B','-m','tools.reports.objective_acceptance_successor',
        '--config',str(Path(template).resolve()),'--local',str(Path(local).resolve()),'--action','run']
    with (root/'driver.log').open('x') as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=child.pid,argv=argv,started_unix=time.time(),
        process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19])
    commit_json(root/'started.json',record);return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--local',required=True);p.add_argument('--action',choices=['validate','run','detach'],default='validate')
    args=p.parse_args();c,prov=load_config(args.config,args.local)
    if args.action=='detach':result=detach(args.config,args.local)
    elif args.action=='run':result=run(c,prov)
    else:
        validate(c);result=dict(status='successor_validated_not_started',npu_executed=False)
    print(json.dumps(result,indent=2))
