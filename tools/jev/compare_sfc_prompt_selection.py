#!/usr/bin/env python3
"""Reaccept complete selection shards and recompute paired strict prompt ranking.

The input spec must be frozen and its hash recorded before selection dispatch.
This tool does not certify that chronology from self-reported timestamps.
"""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash, atomic_json
from multimodal_jev.sfc_jev_shards import population, verify_plan
from multimodal_jev.sfc_prompt_selection import compare
from merge_sfc_jev_shards import collect


def rules_files():
    """Bind the comparator and semantic scoring dependencies before dispatch."""
    root=Path(__file__).parents[2]
    names=['tools/jev/compare_sfc_prompt_selection.py','tools/jev/run_sfc_jev_shard_queue.py']
    names += ['src/multimodal_jev/'+name+'.py' for name in (
        'sfc_prompt_selection','sfc_jev_score','sfc','sfc_jev',
        'sfc_definitions','consumption','sfc_consumption','sfc_omni','sfc_audio',
        'baseline','task_schema')]
    return {name:file_hash(root/name) for name in names}


def load_spec(spec_path, spec_hash):
    if file_hash(spec_path)!=spec_hash:raise ValueError('registered selection spec changed')
    spec=json.loads(spec_path.read_text())
    if set(spec)!={'schema','plan','plan_sha256','roster','prompts','rules_files'} or spec['schema']!='sfc-jev-prompt-comparison-v1':
        raise ValueError('exact frozen comparison specification required')
    if spec['rules_files']!=rules_files():raise ValueError('preregistered selection rules changed')
    plan_path=Path(spec['plan'])
    if file_hash(plan_path)!=spec['plan_sha256']:raise ValueError('selection plan changed')
    plan=json.loads(plan_path.read_text());verify_plan(plan)
    if plan['split']!='train_id' or plan['partition'] is None or plan['partition']['part']!='selection':
        raise ValueError('complete registered selection population required')
    roster=spec['roster']
    if (type(roster) is not list or not 2<=len(roster)<=3 or roster[0]!='parent'
            or roster!=[k for k in ('parent','p1','p2') if k in roster] or set(spec['prompts'])!=set(roster)):
        raise ValueError('full preregistered candidate roster required')
    for pin in spec['prompts'].values():
        if set(pin)!={'roster_path','instruction_sha256','candidate'}:
            raise ValueError('exact pre-dispatch candidate binding required')
    return spec,plan


def dispatch_binding(root, cfg, plan, spec_path, spec_hash):
    """Check a queue against the complete roster before any generation."""
    spec,registered=load_spec(Path(spec_path),spec_hash)
    if (registered!=plan or Path(spec['plan']).resolve()!=Path(cfg['shard_plan']).resolve()
            or spec['plan_sha256']!=cfg['shard_plan_sha256']):
        raise ValueError('queue differs from registered selection plan')
    pin=spec['prompts'].get(cfg['prompt_id'])
    if (pin is None or Path(pin['roster_path']).resolve()!=Path(root).resolve()/'completed-runs.json'
            or pin['instruction_sha256']!=cfg['instruction_sha256'] or pin['candidate']!=cfg.get('candidate')):
        raise ValueError('queue differs from registered selection candidate')
    return dict(path=str(Path(spec_path).resolve()),sha256=spec_hash)


def check_dispatch_record(completed, spec_hash):
    """Require the one-shot queue to have bound this spec before dispatch."""
    root=completed.parent
    intent=json.loads((root/'queue-plan.json').read_text())
    dispatch=json.loads((root/'dispatch-intent.json').read_text())
    if ((intent.get('comparison') or {}).get('sha256')!=spec_hash
            or dispatch.get('comparison_sha256')!=spec_hash
            or dispatch.get('queue_plan_sha256')!=file_hash(root/'queue-plan.json')):
        raise ValueError('selection dispatch did not bind the comparison spec')
    return dict(queue_plan_sha256=file_hash(root/'queue-plan.json'),
                dispatch_intent_sha256=file_hash(root/'dispatch-intent.json'))


def run(spec_path, spec_hash):
    spec,plan=load_spec(spec_path,spec_hash)
    plan_path=Path(spec['plan']);roster=spec['roster']
    all_predictions={};input_rows=None;actors={};evidence={}
    for prompt in roster:
        pin=spec['prompts'][prompt]
        if set(pin)!={'roster_path','instruction_sha256','candidate'}:
            raise ValueError('exact pre-dispatch candidate binding required')
        # Roster path is predeclared; completed bytes cannot exist before dispatch.
        # collect rechecks each exact Docker identity and complete population.
        completed=Path(pin['roster_path']);completed_hash=file_hash(completed)
        dispatch_evidence=check_dispatch_record(completed,spec_hash)
        _,rows,predictions,receipts,actor=collect(plan_path,spec['plan_sha256'],completed,completed_hash)
        cfg=actor['config']
        if (cfg['prompt_id']!=prompt or cfg['instruction_sha256']!=pin['instruction_sha256']
                or cfg.get('candidate')!=pin['candidate']):
            raise ValueError('accepted actor differs from preregistered prompt')
        if input_rows is None:input_rows=rows
        elif input_rows!=rows:raise ValueError('prompt populations differ')
        # Only prompt/context evidence can differ across candidate runs.
        prompt_fields={'prompt_id','instruction_sha256','candidate','context_audit','context_result_sha256',
                       'context_acceptance','context_acceptance_sha256','context_container_id'}
        common=dict(config={k:v for k,v in cfg.items() if k not in prompt_fields},
            source_files=actor['source_files'],model_files=actor['model_files'],versions=actor['versions'])
        if actors and common!=next(iter(actors.values())):
            raise ValueError('model/runtime/source differs between prompt candidates')
        actors[prompt]=common
        all_predictions[prompt]=[{k:p[k] for k in ('example_id','decision_id','request_sha256','raw_text')}
                                 for p in predictions]
        evidence[prompt]=dict(roster_sha256=completed_hash,shards=receipts,dispatch=dispatch_evidence)
    # Only after every prompt's entire population has passed engineering checks.
    release=Path(plan['release']);manifest=json.loads((release/'MANIFEST.json').read_text())
    target=release/'train_id.targets.jsonl'
    if file_hash(target)!=manifest['splits']['train_id']['targets_sha256']:raise ValueError('target release changed')
    wanted={r['id'] for r in input_rows};targets=[]
    with target.open() as handle:
        for row,line in zip(population(release,'train_id'),handle,strict=True):
            if row['id'] not in wanted:continue
            label=json.loads(line)
            if label['id']!=row['id']:raise ValueError('selection target ordering differs')
            targets.append(label)
    result=compare(input_rows,targets,all_predictions,roster)
    return dict(**result,spec_sha256=spec_hash,plan_sha256=spec['plan_sha256'],
        partition=plan['partition'],candidate_evidence=evidence,
        rules_files=spec['rules_files'],
        targets_read='selection rows only; whole TRAIN bytes hashed; no TEST files')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--spec-hash',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve previous comparison')
    report=run(a.spec,a.spec_hash);atomic_json(a.output,report)
    print(json.dumps(report,indent=2))
