"""Formal634 command and collection adapter over the shared isolated lifecycle.

Preparation and launch admission are pending actual efficiency choice and formal
checkpoint/tail acceptance. This module exposes validate/collect, not dispatch.
"""
import argparse
import json
from pathlib import Path

from multimodal_jev.local_config import load_config
from tools.reports.context_macro_training_operations import command as finite_command, validate as shared_validate
from tools.reports.audit_context_macro_training import audit_outputs as shared_audit
from tools.reports.audit_context_macro_probe import collect_and_audit


def command(config, role):
    argv=finite_command(config,role)
    if role=='client':
        old=config['remote_root']+'/package/tools/reports/context_macro_trainer_candidate.py'
        if argv.count(old)!=1:raise ValueError('unique shared training entry required')
        argv[argv.index(old)]=config['remote_root']+'/package/tools/reports/context_macro_formal_candidate.py'
    elif role=='preflight':
        code=argv[-1]
        before='from tools.reports.context_macro_trainer_candidate import generate'
        after='from tools.reports.context_macro_formal_candidate import generate,contract,validate_schedule'
        anchor='assert len(rows)==40513 and len(batches)==634'
        if code.count(before)!=1 or code.count(anchor)!=1:
            raise ValueError('unique shared preflight anchors required')
        code=code.replace(before,after).replace(anchor,
            anchor+'\ncontract(c);validate_schedule(batches,{r["id"] for r in rows})')
        argv[-1]=code
    return argv


def validate(config):
    receipt=shared_validate(config,formal=True,command_builder=command)
    root=Path(config['prepared'])/'package'
    c=json.loads((root/'training.json').read_text())
    if (receipt.get('formal_launch_admitted') is not False
            or c['macro_client']['attempt_id']!='formal-source-v1'
            or receipt.get('selected_microbatch')!=c['microbatch']):
        raise ValueError('explicit undispatched formal candidate with selected microbatch required')
    return receipt


def audit_outputs(package,collection):
    return shared_audit(package,collection,formal=True)


def audit_collected(config):
    from tools.reports.audit_context_macro_training import audit_collected as shared_collect_audit
    return shared_collect_audit(config,validator=validate,command_builder=command,output_auditor=audit_outputs)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    parser.add_argument('--action',choices=['validate','collect','audit'],default='validate')
    args=parser.parse_args();c,_=load_config(args.config,args.local)
    if args.action=='collect':
        result=collect_and_audit(c,validator=validate,command_builder=command,
            output_auditor=audit_outputs,result_dirs=('run','journal','service-0','service-1'))
    elif args.action=='audit':result=audit_collected(c)
    else:result=validate(c)
    print(json.dumps(result,indent=2))
