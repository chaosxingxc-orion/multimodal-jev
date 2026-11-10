#!/usr/bin/env python3
"""Register the complete cross-dataset comparison after independent selection."""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.omni_slurp_transfer_contract import RUNTIME_KEYS
from multimodal_jev.slurp_jev_shards import plan_shards
from select_omni_prompt import select


def register(selection,plan_path,output):
    if json.loads((selection/'evaluation-terminal.json').read_text())['exit_code']!=0:
        raise ValueError('complete successful selection required')
    prior_path=selection/'selection-protocol.json';prior=json.loads(prior_path.read_text())
    result_path=selection/'selection-result.json';result=json.loads(result_path.read_text())
    if select(prior_path,{k:selection/k/'run' for k in prior['prompts']})!=result:
        raise ValueError('full independent selection must reproduce')
    plan=json.loads(plan_path.read_text())
    if (plan['split']!='test' or plan['recordings']!=13078 or plan['decisions']!=2197104
            or plan_shards(plan['release'],'test',plan['recordings_per_shard_target'])!=plan):
        raise ValueError('whole registered TEST shard plan must reproduce')
    identity=json.loads((selection/'parent/run/identity.json').read_text());cfg=dict(identity['config'])
    cfg.setdefault('task_format','jev')
    prompts={'parent':prior['prompts']['parent']};winner=result['selected_id']
    if winner!='parent':prompts[winner]=prior['prompts'][winner]
    report=dict(schema='jev-omni-slurp-transfer-v1',runtime={k:cfg[k] for k in RUNTIME_KEYS},
        prompts=prompts,selection_result=str(result_path.resolve()),selection_result_sha256=file_hash(result_path),
        selection_protocol=str(prior_path.resolve()),selection_protocol_sha256=file_hash(prior_path),
        shard_plan=str(plan_path.resolve()),shard_plan_sha256=file_hash(plan_path),
        asset_audit_sha256=identity['asset_audit_sha256'],versions=identity['versions'],model_files=identity['model_files'],
        image=json.loads((selection/'parent/launch.json').read_text())['image'],
        slurp_feedback_used_for_prompt_optimization=False,
        all_shards_required=True,requests_per_prompt=2197104,weight_updates=0,
        scope='FSC-selected frozen prompt transfer; complete SLURP-Jev TEST, no inference dispatched')
    with output.open('xb') as handle:handle.write(canonical(report))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('selection','plan','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();print(json.dumps(register(a.selection,a.plan,a.output),indent=2))
