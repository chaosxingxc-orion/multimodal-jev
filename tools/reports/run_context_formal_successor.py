"""Wait for exact parity driver, prepare selected formal run, launch and collect.

One attempt, no periodic polling, no model analysis or automatic recovery.
All paths and dispatch choices come from pinned local configuration.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.finalize_context_macro_training import wait_driver
from tools.reports.launch_context_batch_probe_candidate import write_new
from tools.reports.prepare_context_macro_formal import prepare
from tools.reports.launch_context_macro_formal import run as launch
from tools.reports import context_macro_formal_operations as formal
from tools.reports import context_transfer_parity_operations as parity
from tools.reports.audit_context_macro_training import audit_collected as reaudit
from tools.reports.audit_context_macro_probe import collect_and_audit
from tools.reports.context_macro_source_evaluation_plan import build

DATASETS=('FSC_native31','MINDS_native14','SLURP_clean_TEST60')


def read(path):return json.loads(Path(path).read_text())


def pins(c):
    required={c['predecessor_driver'],c['predecessor_config'],c['preparation_config'],
              c['preparation_local'],c['tail_operations']}
    if not required<=set(c['pins']):raise ValueError('all successor inputs must be pinned')
    for name,h in c['pins'].items():
        if file_hash(name)!=h:raise ValueError('formal successor dependency changed: '+name)


def validate(c):
    pins(c);prior=read(c['predecessor_config'])
    preparation,_=load_config(c['preparation_config'],c['preparation_local'])
    tail=read(c['tail_operations'])
    if (c.get('formal_training_dispatch_allowed') is not True
            or c['predecessor_terminal']!=str(Path(prior['output'])/'terminal.json')
            or len(prior['operations'])!=3
            or not set(prior['operations'])<=set(c['pins'])
            or not {preparation['comparison_config'],preparation['comparison_local']}<=set(c['pins'])):
        raise ValueError('exact parity predecessor and explicit formal selection required')
    for name in prior['operations']:
        ops=read(name)
        if ops['host']!=tail['host'] or ops['lease']!=tail['lease']:
            raise ValueError('one serial shared-device lifecycle required')
    if (Path(preparation['output']).exists()
            or (Path(preparation['operations_root'])/'attempt.json').exists()):
        raise ValueError('formal already prepared or attempted; reconcile existing handles')
    return prior,preparation,tail


def run(c,provenance):
    validate(c)
    root=Path(c['output']);check_paths({'formal_successor':root});root.mkdir(parents=True,exist_ok=False)
    write_new(root/'config.json',c);write_new(root/'provenance.json',provenance)
    try:
        predecessor=read(c['predecessor_driver'])
        write_new(root/'waiting.json',dict(process=predecessor,terminal=c['predecessor_terminal']))
        wait_driver(predecessor,Path(c['predecessor_terminal']))
        prior,preparation,tail=validate(c)
        terminal=read(c['predecessor_terminal'])
        if (terminal.get('exit_code')!=0 or terminal.get('status')!='three_target_width_parity_audited'
                or terminal.get('completed')!=list(DATASETS)):
            raise ValueError('all three actual parity datasets must finish successfully')
        policies=[]
        for dataset,name in zip(DATASETS,prior['operations'],strict=True):
            pins(c);ops=read(name)
            proof=reaudit(ops,validator=parity.validate,command_builder=parity.command,
                          output_auditor=parity.audit_outputs)
            if (proof['status']!='target_width_macro_parity_outputs_audited'
                    or proof['comparison']['dataset']!=dataset or proof['comparison']['exact'] is not True):
                raise ValueError('independent actual target-width parity required')
            e=read(Path(ops['prepared'])/'package/evaluation.json')
            policies.append((e['policy_id'],e['contract'],e['base_id']))
            write_new(root/(dataset+'-parity-audit.json'),proof)
        pins(c)
        prepared=prepare(preparation,provenance);write_new(root/'preparation.json',prepared)
        ops=read(Path(preparation['output'])/'operations.json')
        training=read(Path(ops['prepared'])/'package/training.json')
        expected=(training['source_policy_id'],training['source_contract'],training['base_id'])
        if any(p!=expected for p in policies):raise ValueError('parity policy differs from formal common Stage1')
        launch_config=dict(operations=ops,preparation=preparation,tail_operations=tail,
                           formal_training_dispatch_allowed=True)
        write_new(root/'launch-config.json',launch_config)
        pins(c);launch(launch_config,provenance)
        pins(c)
        result=collect_and_audit(ops,validator=formal.validate,command_builder=formal.command,
            output_auditor=formal.audit_outputs,result_dirs=('run','journal','service-0','service-1'))
        if result['status']!='formal_dp2_outputs_audited' or len(result['updates'])!=634:
            raise ValueError('complete formal training output audit required')
        write_new(root/'formal-audit.json',result)
        plan=build(ops,result);write_new(root/'source-evaluation-plan.json',plan)
        write_new(root/'terminal.json',dict(exit_code=0,status='formal_source_collected_and_plan_bound',
            updates=634,source_evaluated=False,target_evaluated=False,automatic_resume=False))
        return result
    except BaseException as exc:
        write_new(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False,preserve_handles=True))
        raise


def detach(template,local,driver_root):
    c,_=load_config(template,local);validate(c)
    root=Path(driver_root);check_paths({'driver':root});root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-m','tools.reports.run_context_formal_successor','--config',str(Path(template).resolve()),
        '--local',str(Path(local).resolve()),'--action','run','--driver-root',str(root.resolve())]
    write_new(root/'intent.json',dict(argv=argv,cwd=str(Path.cwd()),automatic_retry=False))
    with (root/'driver.log').open('x') as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_new(root/'spawn.json',dict(pid=child.pid,argv=argv))
    return dict(pid=child.pid,driver_root=str(root),status='spawned_once')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    parser.add_argument('--action',choices=['validate','run','detach'],default='validate')
    parser.add_argument('--driver-root')
    args=parser.parse_args();c,p=load_config(args.config,args.local)
    import torch
    torch.set_num_threads(1)
    if args.action=='detach':
        if not args.driver_root:parser.error('--driver-root required')
        print(json.dumps(detach(args.config,args.local,args.driver_root)))
    elif args.action=='run':
        if args.driver_root:
            proc=Path('/proc')/str(os.getpid())
            write_new(Path(args.driver_root)/'driver.json',dict(pid=os.getpid(),
                start_ticks=(proc/'stat').read_text().split(') ')[1].split()[19],
                cmdline_sha256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest()))
        run(c,p)
    else:validate(c);print('formal successor configuration valid; no dispatch')
