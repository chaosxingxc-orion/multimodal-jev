"""Evaluate an audited source curve policy with the common macro lifecycle."""
import argparse
import json

from multimodal_jev.local_config import load_config
from tools.reports import context_selection_macro_operations as shared
from tools.reports.audit_context_macro_probe import collect_and_audit
from tools.reports.audit_context_macro_training import audit_collected as reaudit

command=shared.command


def validate(config):
    return shared.validate(config,source_curve=True)


def audit_outputs(package,collection):
    return shared.audit_outputs(package,collection,source_curve=True)


def audit_collected(config):
    return reaudit(config,validator=validate,command_builder=command,output_auditor=audit_outputs)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    parser.add_argument('--action',choices=['validate','run','collect','audit'],default='validate')
    args=parser.parse_args();config,provenance=load_config(args.config,args.local)
    if args.action=='run':result=shared.lifecycle(config,provenance,validator=validate,command_builder=command)
    elif args.action=='collect':
        result=collect_and_audit(config,validator=validate,command_builder=command,output_auditor=audit_outputs,
            result_dirs=('evaluation','journal','service-0','service-1'))
    elif args.action=='audit':result=audit_collected(config)
    else:result=validate(config)
    print(json.dumps(result,indent=2))
