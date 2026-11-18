"""Exact predecessor wait -> capacity32 probe ->100 updates ->audit/compare.

One finite attempt per stage, no polling, retries, model calls or automatic
formal promotion. Existing microbatch study owns its current resource window.
"""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.finalize_context_macro_training import wait_driver
from tools.reports.launch_context_batch_probe_candidate import write_new
from tools.reports.context_capacity_probe_operations import namespace
from tools.reports.context_macro_training_operations import validate,command,run as train
from tools.reports.audit_context_macro_training import audit_collected,audit_outputs
from tools.reports.audit_context_macro_probe import collect_and_audit
from tools.reports.compare_context_service_batch import controls,compare


def execute(c,provenance):
    root=Path(c['output']);check_paths({'study':root});root.mkdir(parents=True,exist_ok=False)
    def pinned():
        for name,digest in c['pins'].items():
            if file_hash(name)!=digest:raise ValueError('study dependency changed: '+name)
    try:
        pinned()
        baseline,_=load_config(c['baseline_config'],c['baseline_local'])
        candidate=json.loads(Path(c['training_operations']).read_text())
        probe=json.loads(Path(c['probe_operations']).read_text())
        validate(baseline);validate(candidate);namespace(probe)['validate'](probe)
        left,lc=controls(baseline);right,rc=controls(candidate)
        if left!=right or lc!=[(4,16384)]*2 or rc!=[(32,131072)]*2:
            raise ValueError('confounded capacity comparison')
        predecessor=json.loads(Path(c['predecessor_driver']).read_text())
        terminal=Path(c['predecessor_terminal'])
        write_new(root/'waiting.json',dict(process=predecessor,terminal=str(terminal)))
        write_new(root/'config.json',c);write_new(root/'provenance.json',provenance)
        wait_driver(predecessor,terminal)
        pinned()
        if json.loads(terminal.read_text()).get('exit_code')!=0:
            raise ValueError('predecessor failed; preserve handles and do not dispatch')
        baseline_audit=audit_collected(baseline)
        write_new(root/'baseline-audit.json',baseline_audit)
        parity=namespace(probe)['launch'](probe,provenance)
        write_new(root/'capacity32-audit.json',parity)
        pinned()
        train(candidate,provenance)
        audit=collect_and_audit(candidate,validator=validate,command_builder=command,
            output_auditor=audit_outputs,result_dirs=('run','journal','service-0','service-1'))
        result=compare(baseline,candidate,[baseline_audit,audit])
        write_new(root/'comparison.json',result)
        write_new(root/'terminal.json',dict(exit_code=0,status='service_batch32_100_steps_compared',formal_dispatched=False))
        return result
    except BaseException as exc:
        write_new(root/'terminal.json',dict(exit_code=1,error=repr(exc),restart=False,preserve_handles=True));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local)
    import torch
    torch.set_num_threads(1);execute(c,provenance)
