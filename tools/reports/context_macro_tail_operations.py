"""Isolated two-update tail diagnostic using the fixed DP2 service lifecycle.

Derive only validation and output audit from pinned shared sources. Existing
finite/formal launchers remain unchanged and reject this experiment kind.
"""
import argparse
import inspect
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from tools.reports import context_macro_training_operations as shared
from tools.reports import audit_context_macro_training as auditor
from tools.reports.context_macro_tail_candidate import contract, select

PINS = {
    'context_macro_training_operations.py': 'ebe4d8a540bbfa26b0933e222167f8cc1e9063f3e7791203680de2be7841d5f5',
    'audit_context_macro_training.py': 'cd26e79996b01440ab26cd52ffbb0e9eafb8045928f697ec5805a4a489dc69bc',
}


def derive(module, function, edits, extra):
    path=Path(module.__file__)
    if file_hash(path)!=PINS[path.name]:raise ValueError('shared tail dependency changed')
    source=inspect.getsource(getattr(module,function))
    for before,after in edits:
        if source.count(before)!=1:raise ValueError('unique tail adapter anchor required: '+before)
        source=source.replace(before,after)
    ns=dict(vars(module));ns.update(extra)
    exec(compile(source,'<tail-'+function+'>','exec'),ns)
    return ns[function]


def command(c,role):
    argv=shared.command(c,role)
    if role=='client':
        old=c['remote_root']+'/package/tools/reports/context_macro_trainer_candidate.py'
        if argv.count(old)!=1:raise ValueError('unique shared client entry required')
        argv[argv.index(old)]=c['remote_root']+'/package/tools/reports/context_macro_tail_candidate.py'
    elif role=='preflight':
        before='from tools.reports.context_macro_trainer_candidate import generate'
        anchor='assert len(rows)==40513 and len(batches)==634'
        if argv[-1].count(before)!=1 or argv[-1].count(anchor)!=1:
            raise ValueError('unique shared preflight anchors required')
        argv[-1]=argv[-1].replace(before,
            'from tools.reports.context_macro_tail_candidate import generate,contract,select').replace(
                anchor,anchor+'\ncontract(c);select(batches,{r["id"] for r in rows})')
    return argv


def validate(c):
    fn=derive(shared,'validate',[
        ("else:efficiency=experiment_window(t)=='efficiency'",'else:tail_contract(t);efficiency=False'),
        ("'macro_training_prefix_prepared_only')","'macro_tail_prepared_only')"),
    ],dict(tail_contract=contract))
    return fn(c,command_builder=command)


def audit_outputs(package,collection):
    fn=derive(auditor,'audit_outputs',[
        ("        efficiency=experiment_window(c)=='efficiency'",'        tail=tail_contract(c);efficiency=False'),
        ("    if formal:contract['formal_experiment']=formal_plan",
         "    contract['tail_admission']=tail"),
        ('stop_update=stop,schedule_updates=634,','stop_update=stop,schedule_updates=2,tail_admission=tail,'),
        ('    real_counts=validate_schedule(batches,population.keys()) if formal else [128]*len(batches)',
         "    batches=select_tail(batches,population.keys())\n    real_counts=[65,128]"),
        ("        if not formal and any(s['padding'] for s in planned['slots']):raise ValueError('finite prefix batches must be real queries')",
         '        # Full schedule, padding truth and weights verified by select_tail and update_targets.'),
        ("'finite_dp2_training_audited',updates=summaries", "'finite_dp2_tail_outputs_audited',updates=summaries"),
        ('fresh_process_replay_verified=bool(start),training_dispatch_allowed=False,',
         'fresh_process_replay_verified=False,training_dispatch_allowed=False,tail_admission=tail,'),
        ('Finite DP2 journal/export/activation audit and explicit replay result; no efficacy or sustained speed claim',
         'Two fresh diagnostic updates on real epoch tail/next batch; no formal history, recovery or efficacy claim'),
    ],dict(tail_contract=contract,select_tail=select))
    return fn(package,collection)


def run(c,provenance):
    if c.get('tail_diagnostic_dispatch_allowed') is not True:raise ValueError('tail diagnostic disabled')
    return shared.run_services(dict(c,finite_inference_dispatch_allowed=True),provenance,
        validator=validate,command_builder=command,required_devices=(4,5,6,7))


def audit_collected(c):
    return auditor.audit_collected(c,validator=validate,command_builder=command,output_auditor=audit_outputs)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['validate','run','collect','audit'],default='validate')
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    if a.action=='run':result=run(c,provenance)
    elif a.action=='audit':result=audit_collected(c)
    elif a.action=='collect':
        from tools.reports.audit_context_macro_probe import collect_and_audit
        result=collect_and_audit(c,validator=validate,command_builder=command,
            output_auditor=audit_outputs,result_dirs=('run','journal','service-0','service-1'))
    else:result=validate(c)
    print(json.dumps(result,indent=2))
