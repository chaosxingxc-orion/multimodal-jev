"""Collect an exact terminated formal attempt for sparse checkpoint recovery.

Retain failure identity. Never stop a process, retry a launch, or reinterpret an
interrupted parent as completed training. All four roles must already be exited.
"""
import argparse
import inspect
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from tools.reports import audit_context_macro_probe as shared
from tools.reports.context_macro_formal_operations import validate, command
from tools.reports.context_macro_formal_candidate import STEPS
from tools.reports.context_macro_probe_operations import inspect_runtime
from tools.reports.audit_context_macro_formal_prefix import audit_prefix

COLLECTOR_SHA256='dfde88c3b0ad40df8dd48f2bb9f1b3d2f9fcd09d8ec1da851c04119176da0d68'


def inspect_exited(observed, handle, *, terminal=False):
    inspect_runtime(observed,handle)
    state=observed['State']
    if state['Status']!='exited' or state['Running']:
        raise ValueError('all exact formal containers must be exited before failure collection')
    # ExitCode/OOM remain in collected-inspect.json. Valid committed prefixes
    # can survive later failures; successful preflight is always required.
    if handle.get('role')=='preflight' and (state['ExitCode']!=0 or state['OOMKilled']):
        raise ValueError('successful formal preflight required')


def collect(c):
    step=c['recovery_step']
    if type(step) is not int or step not in STEPS[1:-1]:
        raise ValueError('explicit nonterminal sparse recovery boundary required')
    if file_hash(Path(shared.__file__))!=COLLECTOR_SHA256:
        raise ValueError('shared failure collector dependency changed')
    source=inspect.getsource(shared.collect_and_audit)
    before="    if terminal['exit_code']!=0:raise ValueError('successful driver terminal required; preserve failure files')"
    after="    if type(terminal['exit_code']) is not int or terminal['exit_code'] not in (0,1):raise ValueError('exact terminated driver required')"
    if source.count(before)!=1:raise ValueError('unique failure collection anchor required')
    source=source.replace(before,after)
    before="write_new(base/'audit.json',result)"
    if source.count(before)!=1:raise ValueError('unique prefix report anchor required')
    source=source.replace(before,"write_new(base/'prefix-audit.json',result)")
    ns=dict(vars(shared),inspect_runtime=inspect_exited)
    exec(compile(source,'<interrupted-formal-collector>','exec'),ns)
    def outputs(p,r):
        return dict(audit_prefix(p,r,step),parent_training_complete=False)
    result=ns['collect_and_audit'](c,validator=validate,command_builder=command,
        output_auditor=outputs,
        result_dirs=('run','journal','service-0','service-1'))
    # The original driver terminal and all exact container states are preserved.
    from tools.reports.prepare_context_macro_sparse_resume import checkpoint
    _,_,_,evidence=checkpoint(dict(step=step,transport_sha256=result['transport_sha256']),c)
    from tools.reports.launch_context_batch_probe_candidate import write_new
    write_new(Path(c['operations_root'])/'recovery-prefix-evidence.json',evidence)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    args=parser.parse_args();config,_=load_config(args.config,args.local)
    shared.torch.set_num_threads(1)
    print(json.dumps(collect(config),indent=2))
