"""Prepare one full source selection point, reauditing its formal origin first."""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from tools.reports.context_macro_formal_operations import audit_collected
from tools.reports.context_macro_source_evaluation_plan import build
from tools.reports.prepare_context_selection_macro import prepare as shared_prepare


def prepare(config,provenance):
    if file_hash(config['source_plan'])!=config['source_plan_sha256']:
        raise ValueError('source evaluation plan changed')
    plan=json.loads(Path(config['source_plan']).read_text())
    operations=plan['source_operations'];admission=audit_collected(operations)
    actual=build(operations,admission)
    if {k:v for k,v in plan.items() if k!='config_provenance'}!=actual:
        raise ValueError('source plan differs from independently audited actual policies')
    points=[p for p in actual['checkpoints'] if p['step']==config['source_step']]
    if type(config['source_step']) is not int or len(points)!=1:
        raise ValueError('one predeclared source evaluation point required')
    if (config['parent']!=operations['prepared'] or config['parent_manifest_sha256']!=operations['manifest_sha256']
            or json.loads(Path(config['parent_audit']).read_text())!=admission):
        raise ValueError('evaluation parent must match freshly audited formal source')
    return shared_prepare(config,provenance,source_point=points[0],source_plan=actual)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    args=parser.parse_args();config,provenance=load_config(args.config,args.local)
    import torch
    torch.set_num_threads(1)
    print(json.dumps(prepare(config,provenance),indent=2))
