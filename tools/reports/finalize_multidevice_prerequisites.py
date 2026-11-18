"""Publish the already completed Stage1 curve using a shared CPU audit worker.

No model inference or training; reuse immutable point audit configurations.
"""
import argparse
import json
from pathlib import Path
from types import FunctionType,SimpleNamespace

from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json
from tools.reports.context_isolated_source_audit import IsolatedSourceAuditEngine
from tools.reports import context_audit_optimized_pipeline as optimized
from tools.reports import collect_context_replica_infonce_curve as curve
from tools.reports.prepare_objective_training_candidate import read_bound


def run(c,provenance):
    prior=json.loads(read_bound(c['curve_config']))
    root=Path(c['output']);root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'config.json',c);commit_json(root/'provenance.json',provenance)
    spec=dict(prior,output=str(root/'curve'))
    commit_json(root/'curve-config.json',spec)
    try:
        with IsolatedSourceAuditEngine(cpu_workers=c['cpu_workers'],io_workers=c['io_workers']) as engine:
            _,point=optimized.optimized_selection(engine)
            reconstruct=FunctionType(curve.reconstruct.__code__,dict(curve.reconstruct.__globals__,
                point=point,reconstruct_plan=engine.reconstruct))
            result=curve.publication.collect(spec,provenance,reconstructor=reconstruct)
        commit_json(root/'terminal.json',dict(exit_code=0,status=result['status'],npu_executed=False))
        return result
    except BaseException as e:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(e),restart=False));raise


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local);print(json.dumps(run(c,provenance),indent=2))
