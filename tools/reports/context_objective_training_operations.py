"""Two-update objective acceptance using the shared four-card lifecycle.

Reconstruct reference admission, bind fixed-image bootstrap proof, then use
the existing device lease, exact container wait, collection and output audit.
No formal-training or recovery admission is granted here.
"""
import argparse
import json
from pathlib import Path
from types import FunctionType

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports import context_cached_training_operations as cached
from tools.reports.prepare_objective_training_candidate import audit as audit_package,read_bound
from tools.reports.context_objective_storage import command
from tools.reports.audit_objective_training import audit as audit_training


def read(path):return json.loads(Path(path).read_text())


def bootstrap_evidence(c):
    report=json.loads(read_bound(c['bootstrap_cpu_report']))
    package=Path(c['training_package']['output']);manifest=read(package/'PREPARED.json')
    if (report['counts']!=dict(tests=18,failures=0,errors=0,skipped=0)
            or report['image']!=manifest['image'] or report['state']['ExitCode']!=0
            or report['state']['OOMKilled'] or report['terminal']['exit_code']!=0):
        raise ValueError('successful fixed-image bootstrap proof required')
    snapshot=Path(c['bootstrap_cpu_source']);files=read(snapshot/'FILES.json')
    if file_hash(snapshot/'FILES.json')!=report['source_manifest_sha256']:
        raise ValueError('bootstrap tested snapshot differs')
    extension=read(package/'config.json')['objective_extension']['files']
    for name,h in extension.items():
        source='tools/reports/context_objective_bootstrap.py' if name=='bootstrap.py' else name
        if files.get(source)!=h or file_hash(snapshot/source)!=h:
            raise ValueError('bootstrap CPU evidence used different extension: '+name)
    return dict(report_sha256=c['bootstrap_cpu_report']['sha256'],tests=18,
                source_manifest_sha256=report['source_manifest_sha256'])


def admission(c):
    spec=c['training_package'];proof=audit_package(spec,{'source':'objective_lifecycle'})
    if proof['status']!='objective_training_candidate_reconstructed_not_admitted':
        raise ValueError('reconstructed objective package required')
    cpu=bootstrap_evidence(c)
    # Preserve all original same-endpoint/reference/forward checks against the
    # candidate files. Only the package schema differs from the old preparer.
    view=dict(spec,parent=spec['parent']['parent'],reference_job=spec['parent']['reference_job'])
    checker=FunctionType(cached.admission.__code__,dict(cached.admission.__globals__,
        audit_package=lambda *_:dict(proof,actor_updates=2)))
    prior=checker(dict(c,training_package=view))
    return dict(status='objective_two_update_prerequisites_audited',
        cached_prerequisites=prior,bootstrap_cpu=cpu,training_dispatch_allowed=False)


def consumer_command(c,m,config,**_):return command(c,m,config)


def validate(c):
    checker=FunctionType(cached.validate.__code__,dict(cached.validate.__globals__,
        admission=admission,command=command,consumer_command=consumer_command))
    return checker(c)


def launch(c,provenance):
    # Cached launch retains shared lease, one-attempt guard and exact collector.
    runner=FunctionType(cached.launch.__code__,dict(cached.launch.__globals__,
        validate=validate,command=command))
    result=runner(c,provenance)
    return dict(result,status='collected_independent_objective_training_audit_required')


def run(c,provenance):
    root=Path(c['operations_root'])/'finalization'
    from multimodal_jev.local_runtime_storage import check_paths
    check_paths({'objective_finalization':root});root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'config.json',c);commit_json(root/'provenance.json',provenance)
    try:
        result=launch(c,provenance)
        if result['status']!='collected_independent_objective_training_audit_required':
            raise ValueError('complete objective collection required')
        stem=Path(c['operations_root'])/c['tag']
        cfg=dict(training_package=c['training_package'],deployment=c,
            objective_loss_atol=c['objective_loss_atol'],
            **{key:str(stem)+suffix for key,suffix in (
                ('launch','-launch.json'),('inspect','-inspect.json'),('wait','-wait.json'),
                ('transport','-collected.transport.json'),('collection','-collected'),('archive','-collected.tar'))})
        commit_json(root/'audit-config.json',cfg)
        proof=audit_training(cfg,provenance);commit_json(root/'audit.json',proof)
        commit_json(root/'terminal.json',dict(exit_code=0,status='objective_prefix_collected_and_audited',
            formal_training_admitted=False,recovery_admitted=False))
        return proof
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False,preserve_handles=True))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--local',required=True);p.add_argument('--action',choices=['validate','run'],default='validate')
    args=p.parse_args();c,provenance=load_config(args.config,args.local)
    if args.action=='run':result=run(c,provenance)
    else:
        m,config,_=validate(c)
        result=dict(status='objective_lifecycle_validated_not_executed',argv=command(c,m,config),
                    npu_executed=False,training_dispatch_allowed=False)
    print(json.dumps(result,indent=2))
