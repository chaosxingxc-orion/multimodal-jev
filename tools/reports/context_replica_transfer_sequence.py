"""Exact GRPO source successor: full native inference, then frozen-role scoring.

No training, retries, target-driven policy choice or periodic model calls.
Each unique policy runs once per dataset; duplicate roles share its collection.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.finalize_context_infonce import wait_exact
from tools.reports.context_release_deployment import inherited
from tools.reports import context_cached_grpo_evaluation_sequence as predecessor
from tools.reports.context_cached_grpo_transfer_source import admit
from tools.reports import prepare_context_replica_transfer as preparation
from tools.reports import context_replica_transfer_operations as operations
from tools.reports import collect_context_replica_transfer_scores as scoring
from tools.reports.score_context_transfer_macro import ROLES
from tools.reports.context_replica_training_operations import path

DATASETS = ('FSC_native31', 'MINDS_native14', 'SLURP_clean_TEST60')
SLUGS = ('fsc', 'minds', 'slurp-test')


def read(p):
    return json.loads(Path(p).read_text())


def pins(c):
    for name, expected in c['pins'].items():
        if file_hash(name) != expected:
            raise ValueError('transfer sequence dependency changed: '+name)
    prior = c['predecessor']
    for key in ('driver', 'config'):
        if file_hash(prior[key]) != prior[key+'_sha256']:
            raise ValueError('source predecessor changed: '+key)


def validate(c):
    if c.get('transfer_dispatch_allowed') is not True or set(c['datasets']) != set(DATASETS):
        raise ValueError('explicit three simple native datasets required')
    pins(c)
    prior = c['predecessor']
    previous = read(prior['config'])
    base = Path(previous['output'])
    expected = dict(config=base/'config.json', terminal=base/'terminal.json',
        driver=base.with_name(base.name+'-driver')/'started.json')
    if any(prior[k] != str(v) for k,v in expected.items()):
        raise ValueError('exact source evaluation predecessor paths required')
    formal, selection = predecessor.validate(previous)
    # Protect the complete existing chain, including not-yet-created packages.
    _, efficiency = predecessor.predecessor.validate(formal)
    finite = predecessor.predecessor.selection.efficiency_sequence.validate(efficiency)
    reference = read(finite['reference_predecessor']['config'])
    protected = []
    def paths(value):
        if isinstance(value, dict):
            for k,v in value.items():
                if k in ('output','package','operations_root','remote_root','root','source_plan','lease') and isinstance(v,str):
                    protected.append(Path(v).resolve())
                elif isinstance(v,(dict,list)):
                    paths(v)
        elif isinstance(value,list):
            for v in value: paths(v)
    for v in (previous, formal, selection, efficiency, finite, reference): paths(v)
    protected.extend([base.resolve(), base.with_name(base.name+'-driver').resolve()])
    for dataset, item in c['datasets'].items():
        if (set(item) != {'portable_full_inputs','portable_parity_inputs','original_evaluation','scoring'}
                or item['original_evaluation']['dataset'] != dataset
                or item['portable_full_inputs']['cache_root'] != item['portable_parity_inputs']['cache_root']):
            raise ValueError('fixed native input/scoring configuration required')
        for key in ('portable_full_inputs','portable_parity_inputs'):
            spec=item[key]
            manifest=Path(path(spec['root']))/'MANIFEST.json'
            if file_hash(manifest) != spec['manifest_sha256']:
                raise ValueError('portable input identity changed')
            protected.extend([Path(spec['root']).resolve(),Path(path(spec['cache_root'])).resolve()])
        paths(item['original_evaluation'])
    roots=[Path(path(c[k])).resolve() for k in ('output','prepared_root','operations_root','scores_root','remote_root')]
    roots.append(roots[0].with_name(roots[0].name+'-driver'))
    for i,p in enumerate(roots):
        if any(p.is_relative_to(q) or q.is_relative_to(p) for q in roots[:i]+protected):
            raise ValueError('new transfer paths must be disjoint from each other and all dependencies')
    if (not re.fullmatch('[a-z0-9][a-z0-9-]*', c['tag_prefix'])
            or len(str(roots[4]/'slurp-test-role-2/service-inference1/socket').encode()) > 107):
        raise ValueError('safe tag and bounded socket path required')
    check_paths({k:c[k] for k in ('output','prepared_root','operations_root','scores_root')})
    return previous, selection


def wait_source(c):
    previous, selection = validate(c)
    prior=c['predecessor']
    wait_exact(read(prior['driver']), prior['terminal'])
    if validate(c) != (previous, selection):
        raise ValueError('source predecessor changed while waiting')
    terminal=read(prior['terminal'])
    if (terminal.get('exit_code') != 0 or terminal.get('status') != 'cached_grpo_source_curve_collected_and_audited'
            or terminal.get('points') != 11 or terminal.get('target_inference_performed') is not False):
        raise ValueError('complete independently audited source curve required')
    base=Path(previous['output'])
    curve=read(base/'curve-config.json')
    result=read(base/'curve/result.json')
    if (curve['output'] != str(base/'curve') or curve['source_plan'] != str(base/'source-plan.json')
            or curve['source_plan_sha256'] != file_hash(base/'source-plan.json')
            or result['status'] != 'complete_source_curve_audited_and_roles_frozen'
            or result['source_evaluation'] != dict(path=str(base/'curve/source-evaluation.json'),
                sha256=file_hash(base/'curve/source-evaluation.json'))
            or set(result['roles']) != set(ROLES)):
        raise ValueError('actual predecessor curve publication required')
    if [e['step'] for e in curve['evaluations']] != [e['step'] for e in previous['points']]:
        raise ValueError('exact registered predecessor points required')
    for entry, point in zip(curve['evaluations'], previous['points'], strict=True):
        audit_path=Path(point['operations_root'])/'finalization/audit-config.json'
        if entry['audit_config'] != str(audit_path) or entry['audit_config_sha256'] != file_hash(audit_path):
            raise ValueError('actual predecessor point collections required')
        bound=read(audit_path)
        if (bound['selection_package']['output'] != point['package']
                or bound['selection_package']['remote_root'] != point['remote_root']
                or bound['deployment']['tag'] != point['tag']):
            raise ValueError('predecessor point execution differs')
    bindings={}
    common=None
    for role in ROLES:
        request=dict(source_evaluation=result['source_evaluation'],source_curve_config=curve,
            policy_role=role,policy_id=result['roles'][role]['policy_id'])
        proof=admit(request)  # Reconstruct original training, all selection, tensors.
        if proof['binding'] != result['roles'][role]:
            raise ValueError('published source role differs from reconstruction')
        identity=(proof['source_operations'],proof['training'])
        if common is None: common=identity
        if common != identity:
            raise ValueError('same source training for every role required')
        bindings[role]=request
    bound, _ = common
    parent=bound['training_package']['parent']
    deployment=inherited(bound['deployment'])
    if parent != selection['point_template']['parent'] or deployment != selection['deployment']:
        raise ValueError('same actual parent, host and four-card lease required')
    return bindings,parent,deployment


def run(c, provenance):
    validate(c)
    root=Path(c['output'])
    for key in ('output','prepared_root','operations_root','scores_root'):
        if Path(c[key]).exists(): raise FileExistsError(c[key])
    root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'config.json',c)
    commit_json(root/'provenance.json',provenance)
    try:
        commit_json(root/'waiting.json',c['predecessor'])
        bindings,parent,deployment=wait_source(c)
        commit_json(root/'source-bindings.json',bindings)
        matrices={}
        for dataset,slug in zip(DATASETS,SLUGS,strict=True):
            item=c['datasets'][dataset]
            executions={};entries={}
            for index,role in enumerate(ROLES):
                pins(c)
                binding=bindings[role];policy=binding['policy_id']
                if policy not in executions:
                    name=slug+f'-role-{index}'
                    spec=dict(parent=parent,source_admission=binding,dataset=dataset,
                        portable_full_inputs=item['portable_full_inputs'],portable_parity_inputs=item['portable_parity_inputs'],
                        output=str(Path(c['prepared_root'])/name),remote_root=str(Path(c['remote_root'])/name))
                    if dataset=='SLURP_clean_TEST60': spec['source_release']=item['original_evaluation']['source_release']
                    spec['manifest_sha256']=preparation.prepare(spec,provenance)['manifest_sha256']
                    ops=dict(deployment,transfer_package=spec,package=spec['output'],manifest_sha256=spec['manifest_sha256'],
                        operations_root=str(Path(c['operations_root'])/name),tag=c['tag_prefix']+'-'+name,transfer_dispatch_allowed=True)
                    commit_json(root/(name+'-operations.json'),ops)
                    result=operations.run(ops,provenance)
                    if (result['status']!='complete_replica_transfer_audited' or result['runtime_provenance_verified'] is not True
                            or result['dataset']!=dataset or result['policy_id']!=policy or result['target_scored'] is not False):
                        raise ValueError('actual complete inference admission required')
                    audit_path=Path(ops['operations_root'])/'finalization/audit-config.json'
                    executions[policy]=read(audit_path)
                    if executions[policy]['deployment']!=ops or executions[policy]['transfer_package']!=spec:
                        raise ValueError('actual finalization must match dispatched package')
                    commit_json(root/(name+'-completed.json'),dict(audit_config=str(audit_path),
                        audit_config_sha256=file_hash(audit_path),policy_id=policy,dataset=dataset))
                entries[role]=dict(policy_id=policy,operations=executions[policy])
            matrices[dataset]=dict(entries=entries,original_evaluation=item['original_evaluation'],
                scoring=item['scoring'],output=str(Path(c['scores_root'])/slug))
        # No target scores can affect which policy/dataset is dispatched.
        results={}
        for dataset,slug in zip(DATASETS,SLUGS,strict=True):
            pins(c)
            config=matrices[dataset]
            commit_json(root/(slug+'-scoring-config.json'),config)
            result=scoring.collect(config,provenance)
            if result['dataset']!=dataset or result['runtime_provenance_verified'] is not True:
                raise ValueError('complete native score matrix admission required')
            p=Path(config['output'])/'scores.json'
            results[dataset]=dict(path=str(p),sha256=file_hash(p))
        commit_json(root/'results.json',dict(datasets=results,target_tuning=False,source_bindings=bindings))
        commit_json(root/'terminal.json',dict(exit_code=0,status='replica_three_dataset_role_matrices_scored',
            datasets=list(DATASETS),training_dispatched=False,target_tuning=False,restart=False))
        return results
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False,preserve_handles=True))
        raise


def detach(template, local):
    c,_=load_config(template,local)
    validate(c)
    root=Path(c['output']).with_name(Path(c['output']).name+'-driver')
    check_paths({'transfer_driver':root})
    root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-B','-m','tools.reports.context_replica_transfer_sequence','--config',str(Path(template).resolve()),
        '--local',str(Path(local).resolve()),'--action','run']
    with (root/'driver.log').open('x') as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=child.pid,argv=argv,started_unix=time.time(),
        process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19])
    commit_json(root/'started.json',record)
    return record


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['validate','run','detach'],default='validate')
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    if a.action=='detach': result=detach(a.config,a.local)
    elif a.action=='run': result=run(c,provenance)
    else:
        validate(c)
        result=dict(status='replica_transfer_sequence_validated_only',npu_executed=False)
    print(json.dumps(result,indent=2))
