"""Frozen source roles to full native target inference and final scores.

Wait for the exact source-curve driver. Execute each unique policy once per
dataset, reuse aliases, and finish all inference before native target scoring.
No target tuning, model analysis, training, retries or automatic recovery.
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
from tools.reports.context_transfer_source_admission import admit
from tools.reports.prepare_context_full_transfer import prepare
from tools.reports import context_full_transfer_operations as full
from tools.reports.collect_context_full_transfer_scores import collect as score
from tools.reports.score_context_transfer_macro import ROLES

DATASETS=('FSC_native31','MINDS_native14','SLURP_clean_TEST60')
SLUGS=('fsc','minds','slurp-test')


def read(p):return json.loads(Path(p).read_text())


def pins(c):
    if not {c['predecessor_driver'],c['predecessor_config']}<=set(c['pins']):
        raise ValueError('pinned source-curve predecessor required')
    for p,h in c['pins'].items():
        if file_hash(p)!=h:raise ValueError('full transfer dependency changed: '+p)


def validate(c):
    pins(c);prior=read(c['predecessor_config'])
    if (c['predecessor_terminal']!=str(Path(prior['output'])/'terminal.json')
            or c.get('full_transfer_dispatch_allowed') is not True or set(c['datasets'])!=set(DATASETS)):
        raise ValueError('exact complete source predecessor and three fixed datasets required')
    roots=[Path(c[k]) for k in ('output','prepared_root','operations_root','scores_root')]
    if any(not p.is_absolute() or p.exists() for p in roots):raise ValueError('new absolute local output roots required')
    if any(a.resolve().is_relative_to(b.resolve()) or b.resolve().is_relative_to(a.resolve())
           for i,a in enumerate(roots) for b in roots[i+1:]):raise ValueError('disjoint local output roots required')
    remote=Path(c['remote_root'])
    if (not remote.is_absolute() or '..' in remote.parts or ':' in str(remote)
            or len(str(remote/'slurp-test-role-2/ipc').encode())+8>107):
        raise ValueError('isolated bounded remote transfer root required')
    for dataset,item in c['datasets'].items():
        if (set(item)!={'portable_full_inputs','portable_parity_inputs','original_evaluation','scoring'}
                or item['original_evaluation']['dataset']!=dataset):
            raise ValueError('fixed target input/scoring configuration only')
    check_paths({k:c[k] for k in ('output','prepared_root','operations_root','scores_root')})
    return prior


def run(c,provenance):
    prior=validate(c);root=Path(c['output']);root.mkdir(parents=True,exist_ok=False)
    write_new(root/'config.json',c);write_new(root/'provenance.json',provenance)
    try:
        predecessor=read(c['predecessor_driver'])
        write_new(root/'waiting.json',dict(process=predecessor,terminal=c['predecessor_terminal']))
        wait_driver(predecessor,Path(c['predecessor_terminal']));pins(c)
        terminal=read(c['predecessor_terminal'])
        if terminal.get('exit_code')!=0 or terminal.get('status')!='source_curve_completed_roles_frozen':
            raise ValueError('successful complete source curve required')
        published=read(Path(prior['curve_output'])/'result.json')
        curve_config=read(Path(prior['output'])/'curve-config.json')
        if (published['status']!='complete_source_curve_audited_and_roles_frozen'
                or published['source_evaluation']!=terminal['source_evaluation']
                or curve_config['output']!=prior['curve_output']
                or set(published['roles'])!=set(ROLES)):
            raise ValueError('same published complete source roles required')
        bindings={};common=None
        for role in ROLES:
            request=dict(source_evaluation=published['source_evaluation'],source_curve_config=curve_config,
                policy_role=role,policy_id=published['roles'][role]['policy_id'])
            proof=admit(request)
            if proof['binding']!=published['roles'][role]:raise ValueError('published source role differs from reconstruction')
            identity=(proof['source_operations'],proof['training'],proof['source_evaluation'])
            if common is None:common=identity
            if common!=identity:raise ValueError('roles must originate in the same source training')
            bindings[role]=request
            write_new(root/(role+'-source-admission.json'),proof)
        matrices={}
        for dataset,slug in zip(DATASETS,SLUGS,strict=True):
            item=c['datasets'][dataset];executions={};entries={}
            for index,role in enumerate(ROLES):
                pins(c);binding=bindings[role];policy=binding['policy_id']
                if policy not in executions:
                    name=slug+f'-role-{index}'
                    prep=dict(project=c['project'],source_admission=binding,
                        portable_full_inputs=item['portable_full_inputs'],portable_parity_inputs=item['portable_parity_inputs'],
                        output=str(Path(c['prepared_root'])/name),remote_root=str(Path(c['remote_root'])/name),
                        operations_root=str(Path(c['operations_root'])/name),tag=c['tag_prefix']+'-'+name)
                    write_new(root/(name+'-preparation.json'),prep);prepare(prep,provenance)
                    ops=read(Path(prep['output'])/'operations.json');ops['finite_inference_dispatch_allowed']=True
                    write_new(root/(name+'-operations.json'),ops)
                    full.validate(ops)
                    full.lifecycle(ops,provenance,validator=full.validate,command_builder=full.command)
                    result=full.collect(ops)
                    if (result['status']!='full_portable_transfer_outputs_audited'
                            or result['runtime_provenance_verified'] is not True
                            or result['policy_id']!=policy or result['dataset']!=dataset):
                        raise ValueError('full executed target runtime audit required')
                    write_new(root/(name+'-audit.json'),result);executions[policy]=ops
                entries[role]=dict(policy_id=policy,operations=executions[policy])
            matrices[dataset]=dict(entries=entries,original_evaluation=item['original_evaluation'],
                scoring=item['scoring'],output=str(Path(c['scores_root'])/slug))
        # All requested inference has finished; scores cannot influence dispatch.
        results={}
        for dataset,slug in zip(DATASETS,SLUGS,strict=True):
            pins(c);scoring=matrices[dataset]
            write_new(root/(slug+'-scoring-config.json'),scoring)
            result=score(scoring,provenance)
            if result['dataset']!=dataset or result['runtime_provenance_verified'] is not True:
                raise ValueError('full native score matrix admission required')
            results[dataset]=dict(path=str(Path(scoring['output'])/'scores.json'),
                sha256=file_hash(Path(scoring['output'])/'scores.json'))
        write_new(root/'results.json',dict(datasets=results,target_tuning=False,source_evaluation=published['source_evaluation']))
        write_new(root/'terminal.json',dict(exit_code=0,status='full_three_dataset_role_matrices_scored',
            datasets=list(DATASETS),target_tuning=False,training_dispatched=False))
        return results
    except BaseException as exc:
        write_new(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False,preserve_handles=True));raise


def detach(template,local,driver_root):
    c,_=load_config(template,local);validate(c)
    root=Path(driver_root);check_paths({'driver':root});root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-m','tools.reports.run_context_full_transfer_successor','--config',str(Path(template).resolve()),
          '--local',str(Path(local).resolve()),'--action','run','--driver-root',str(root.resolve())]
    write_new(root/'intent.json',dict(argv=argv,cwd=str(Path.cwd()),automatic_retry=False))
    with (root/'driver.log').open('x') as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_new(root/'spawn.json',dict(pid=child.pid,argv=argv))
    return dict(pid=child.pid,driver_root=str(root),status='spawned_once')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    parser.add_argument('--action',choices=['validate','run','detach'],default='validate');parser.add_argument('--driver-root')
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
    else:validate(c);print('full transfer successor valid; no dispatch')
