"""Sparse formal recovery over the shared DP2 and dual TP1 lifecycle.

Revalidate the exact parent prefix before deployment. Preserve frozen runtime
bytes, isolated local storage, device lease and one-attempt semantics.
"""
import argparse
import inspect
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from tools.reports import context_macro_training_operations as shared
from tools.reports import audit_context_macro_training as collector
from tools.reports.audit_context_macro_probe import collect_and_audit
from tools.reports.audit_context_macro_formal_recovery import bind_prefix, audit_outputs
from tools.reports.context_macro_formal_resume import resume_contract

VALIDATOR_SHA256='ebe4d8a540bbfa26b0933e222167f8cc1e9063f3e7791203680de2be7841d5f5'


def command(c, role):
    argv=shared.command(c,role)
    if role=='client':
        old=c['remote_root']+'/package/tools/reports/context_macro_trainer_candidate.py'
        if argv.count(old)!=1:raise ValueError('unique recovery client entry required')
        argv[argv.index(old)]=c['remote_root']+'/package/tools/reports/context_macro_formal_resume.py'
    elif role=='preflight':
        edits=[
            ('from tools.reports.context_macro_trainer_candidate import generate',
             'from tools.reports.context_macro_formal_resume import generate,validate_resume\n'
             'from tools.reports.context_macro_formal_candidate import validate_schedule'),
            ('assert len(rows)==40513 and len(batches)==634',
             'assert len(rows)==40513 and len(batches)==634\n'
             'validate_resume(c);validate_schedule(batches,{r["id"] for r in rows})'),
            ("verify_native_adapter(c['initial_adapter'],b)",
             "verify_native_adapter(c['reference_adapter'],b)"),
        ]
        for before,after in edits:
            if argv[-1].count(before)!=1:raise ValueError('unique recovery preflight anchor required')
            argv[-1]=argv[-1].replace(before,after)
    return argv


def validate(c):
    package=Path(c['prepared'])/'package'
    receipt=json.loads((package.parent/'PREPARED.json').read_text())
    t=json.loads((package/'training.json').read_text());step=t['start_update']
    if (receipt.get('prefix_learning_history_verified') is not True
            or receipt.get('formal_launch_admitted') is not False
            or receipt.get('step')!=step
            or receipt.get('selected_microbatch')!=t['microbatch']
            or receipt.get('recovery_sha256')!=file_hash(package/'RECOVERY.json')
            or t['macro_client']['attempt_id']!=f'formal-resume-{step:04d}'):
        raise ValueError('sealed sparse recovery metadata required')
    if file_hash(Path(shared.__file__))!=VALIDATOR_SHA256:
        raise ValueError('frozen shared recovery validator changed')
    source=inspect.getsource(shared.validate)
    edits=[
        ('        contract(t)', '        resume_contract(t)'),
        ("replay=t['start_update']==1", "replay=True"),
        ("package/'REPLAY.json'", "package/'RECOVERY.json'"),
        ("r.get('replay_sha256')", "r.get('recovery_sha256')"),
        ("root+'/package/resume/policy-0001'", "root+f'/package/resume/policy-{t[\"start_update\"]:04d}'"),
        ("root+'/package/resume/adapter-0001'", "root+f'/package/resume/adapter-{t[\"start_update\"]:04d}'"),
        ("'macro_formal_prepared_only'", "'macro_sparse_resume_prepared_only'"),
    ]
    # REPLAY path appears twice (read and hash); all other anchors are unique.
    for before,after in edits:
        if source.count(before)!=(2 if before=="package/'REPLAY.json'" else 1):
            raise ValueError('shared recovery validation anchor changed: '+before)
        source=source.replace(before,after)
    ns=dict(vars(shared),resume_contract=resume_contract)
    exec(compile(source,'<sparse-recovery-validator>','exec'),ns)
    accepted=ns['validate'](c,formal=True,command_builder=command)
    _,bound_step=bind_prefix(package)
    if bound_step!=step:raise ValueError('recovery prefix cursor differs')
    return accepted


def run(c,provenance):
    if c.get('formal_recovery_dispatch_allowed') is not True:
        raise ValueError('formal recovery dispatch disabled')
    return shared.run_services(dict(c,finite_inference_dispatch_allowed=True),provenance,
        validator=validate,command_builder=command,required_devices=(4,5,6,7))


def audit_collected(c):
    package=Path(c['prepared'])/'package'
    binding=json.loads((package/'RECOVERY.json').read_text())
    observations=json.loads((Path(c['operations_root'])/'collected-inspect.json').read_text())
    if set(observations)!={'preflight','client','service-0','service-1'}:
        raise ValueError('complete recovery container evidence required')
    if set(v['Id'] for v in observations.values()) & set(binding['containers'].values()):
        raise ValueError('recovery requires fresh containers')
    result=collector.audit_collected(c,validator=validate,command_builder=command,
                                    output_auditor=audit_outputs)
    result.update(resumed_runtime_transport_verified=True)
    return result


def collect(c):
    collect_and_audit(c,validator=validate,command_builder=command,
        output_auditor=audit_outputs,result_dirs=('run','journal','service-0','service-1'))
    result=audit_collected(c)
    from tools.reports.launch_context_batch_probe_candidate import write_new
    write_new(Path(c['operations_root'])/'recovery-audit.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    parser.add_argument('--action',choices=['validate','run','collect','audit'],default='validate')
    args=parser.parse_args();config,provenance=load_config(args.config,args.local)
    collector.torch.set_num_threads(1)
    if args.action=='run':result=run(config,provenance)
    elif args.action=='collect':result=collect(config)
    elif args.action=='audit':result=audit_collected(config)
    else:result=validate(config)
    print(json.dumps(result,indent=2))
