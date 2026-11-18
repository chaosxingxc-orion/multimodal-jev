"""After exact formal completion, evaluate all11 source points and freeze roles.

Serial full-population inference on the existing dual-service lifecycle. No
target inputs, test-set selection, retries, optimizer updates or model polling.
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
from tools.reports.context_macro_formal_candidate import STEPS
from tools.reports.context_macro_formal_operations import audit_collected as audit_training
from tools.reports.context_macro_source_evaluation_plan import build
from tools.reports.prepare_context_source_point_macro import prepare
from tools.reports import context_source_point_macro_operations as point
from tools.reports.audit_context_macro_probe import collect_and_audit
from tools.reports.collect_context_macro_source_curve import collect as collect_curve

INPUT_KEYS={'project','binding_root','binding_sha256','cache_audit','cache_audit_sha256',
            'cache_collection','cache_root','release','release_sha256'}


def read(path):return json.loads(Path(path).read_text())


def pins(c):
    required={c['predecessor_driver'],c['predecessor_config'],c['operations_template']}
    if not required<=set(c['pins']):raise ValueError('pinned predecessor and evaluation template required')
    for path,h in c['pins'].items():
        if file_hash(path)!=h:raise ValueError('source curve dependency changed: '+path)


def validate(c):
    pins(c);prior=read(c['predecessor_config'])
    if (c['predecessor_terminal']!=str(Path(prior['output'])/'terminal.json')
            or c['source_plan']!=str(Path(prior['output'])/'source-evaluation-plan.json')
            or c.get('source_evaluation_dispatch_allowed') is not True
            or set(c['inputs'])!=INPUT_KEYS):
        raise ValueError('exact formal predecessor and input-only preparation configuration required')
    if read(c['operations_template'])!='${SOURCE_POINT_OPERATIONS}':
        raise ValueError('whole-object local operations macro required')
    root=Path(c['remote_root'])
    if (not root.is_absolute() or '..' in root.parts or ':' in str(root)
            or len(str(root/'step-0634/ipc').encode())+8>107):
        raise ValueError('bounded absolute remote evaluation root required')
    for key in ('output','prepared_root','operations_root','curve_output'):
        p=Path(c[key])
        if not p.is_absolute() or p.exists():raise ValueError('fresh local curve paths required: '+key)
    paths=[Path(c[k]).resolve() for k in ('output','prepared_root','operations_root','curve_output')]
    if any(a.is_relative_to(b) or b.is_relative_to(a) for i,a in enumerate(paths) for b in paths[i+1:]):
        raise ValueError('disjoint source curve output paths required')
    check_paths({k:c[k] for k in ('output','prepared_root','operations_root','curve_output')})
    return prior


def run(c,provenance):
    prior=validate(c);root=Path(c['output']);root.mkdir(parents=True,exist_ok=False)
    write_new(root/'config.json',c);write_new(root/'provenance.json',provenance)
    try:
        predecessor=read(c['predecessor_driver'])
        write_new(root/'waiting.json',dict(process=predecessor,terminal=c['predecessor_terminal']))
        wait_driver(predecessor,Path(c['predecessor_terminal']));pins(c)
        terminal=read(c['predecessor_terminal'])
        if (terminal.get('exit_code')!=0 or terminal.get('status')!='formal_source_collected_and_plan_bound'
                or terminal.get('updates')!=634):
            raise ValueError('complete formal source predecessor required')
        plan=read(c['source_plan']);ops=plan['source_operations']
        launched=read(Path(prior['output'])/'launch-config.json')['operations']
        if ops!=launched:raise ValueError('source plan belongs to a different formal attempt')
        training=audit_training(ops);actual=build(ops,training)
        if (plan!=actual or [p['step'] for p in actual['checkpoints']]!=STEPS
                or c['inputs']['release_sha256']!=actual['experiment']['release_sha256']):
            raise ValueError('complete actual source plan and matching clean release required')
        train_config=read(Path(ops['prepared'])/'package/training.json')
        if (c['inputs']['release']!=train_config['release']
                or Path(c['remote_root']).is_relative_to(ops['remote_root'])
                or Path(ops['remote_root']).is_relative_to(c['remote_root'])):
            raise ValueError('source population and isolated evaluation runtime required')
        audit_path=root/'formal-audit.json';write_new(audit_path,training)
        entries=[]
        for step in STEPS:
            pins(c);name=f'step-{step:04d}'
            prep=dict(c['inputs'],parent=ops['prepared'],parent_manifest_sha256=ops['manifest_sha256'],
                parent_audit=str(audit_path),parent_audit_sha256=file_hash(audit_path),
                source_plan=c['source_plan'],source_plan_sha256=file_hash(c['source_plan']),source_step=step,
                output=str(Path(c['prepared_root'])/name),remote_root=str(Path(c['remote_root'])/name))
            write_new(root/(name+'-preparation.json'),prep)
            receipt=prepare(prep,provenance)
            p=dict(prepared=prep['output'],manifest_sha256=receipt['files_sha256'],remote_root=prep['remote_root'],
                operations_root=str(Path(c['operations_root'])/name),tag=c['tag_prefix']+'-'+name,
                **{k:ops[k] for k in ('host','model','driver_root','lease')},
                cache_root=c['inputs']['cache_root'],release=c['inputs']['release'],
                release_sha256=c['inputs']['release_sha256'],finite_inference_dispatch_allowed=True)
            local=root/(name+'-operations-local.json')
            write_new(local,dict(schema='jev-local-macros-v1',macros=dict(SOURCE_POINT_OPERATIONS=p)))
            point.validate(p)
            write_new(root/(name+'-dispatch.json'),dict(operations=p,source_step=step))
            point.shared.lifecycle(p,provenance,validator=point.validate,command_builder=point.command)
            proof=collect_and_audit(p,validator=point.validate,command_builder=point.command,
                output_auditor=point.audit_outputs,result_dirs=('evaluation','journal','service-0','service-1'))
            if (proof['status']!='complete_source_point_macro_selection_audited'
                    or proof['result']['policy_id']!=next(q['policy_id'] for q in actual['checkpoints'] if q['step']==step)):
                raise ValueError('complete same-policy source selection score required')
            write_new(root/(name+'-audit.json'),proof)
            entries.append(dict(step=step,config=c['operations_template'],local=str(local)))
        pins(c)
        curve_config=dict(source_plan=c['source_plan'],source_plan_sha256=file_hash(c['source_plan']),
                          evaluations=entries,output=c['curve_output'])
        write_new(root/'curve-config.json',curve_config)
        result=collect_curve(curve_config,provenance)
        if result['status']!='complete_source_curve_audited_and_roles_frozen':
            raise ValueError('complete independently reconstructed source curve required')
        write_new(root/'terminal.json',dict(exit_code=0,status='source_curve_completed_roles_frozen',
            source_evaluation=result['source_evaluation'],target_evaluated=False))
        return result
    except BaseException as exc:
        write_new(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False,preserve_handles=True))
        raise


def detach(template,local,driver_root):
    c,_=load_config(template,local);validate(c)
    root=Path(driver_root);check_paths({'driver':root});root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-m','tools.reports.run_context_source_curve_successor','--config',str(Path(template).resolve()),
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
    else:validate(c);print('source curve successor valid; no dispatch')
