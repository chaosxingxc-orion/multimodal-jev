#!/usr/bin/env python3
"""Pin a prospective compositional-transfer study without starting training."""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.span_rule_transfer import make_transfer_protocol
from multimodal_jev.span_rule_balance import add_complementary_rules,balanced_rule_schedule
from multimodal_jev.policy_identity import checkpoint_identity
from verify_span_rule_controls import verify as verify_parent
from verify_span_selection_rules import verify as verify_selection


def prepare(learning,selection,initializer):
    read=lambda p:json.loads(p.read_text())
    manifest=read(learning/'MANIFEST.json');parent=Path(manifest['parent'])
    parent_audit=verify_parent(parent);selection_audit=verify_selection(selection)
    if manifest['schema']!='jev-span-balanced-rule-controls-v1' or parent_audit['manifest_sha256']!=manifest['parent_manifest_sha256']:
        raise ValueError('learning lineage differs')
    for name,digest in manifest['files'].items():
        if file_hash(learning/name)!=digest:raise ValueError('learning artifact differs')
    for path,digest in manifest['source_sha256'].items():
        if file_hash(path)!=digest:raise ValueError('learning source differs')
    load=lambda p:[json.loads(s) for s in p.read_text().splitlines()]
    rows=load(learning/'learning.jsonl')
    if rows!=add_complementary_rules(load(parent/'learning.jsonl')):
        raise ValueError('learning derivation differs')
    if read(learning/'schedule.json')!=balanced_rule_schedule(rows,manifest['seed']):
        raise ValueError('learning balanced schedule differs')
    report=make_transfer_protocol(rows,load(selection/'selection.jsonl'))
    config=read(initializer/'config.json');result=read(initializer/'run/result.json')
    identity=read(initializer/'run/identity.json');history=read(initializer/'run/history.json')
    if (read(initializer/'run/terminal.json')['exit_code']!=0
            or config['question_order']!=['date','time','timeofday']
            or config['objective']!='candidate_ce' or config.get('initial_adapter')
            or config['lora_execution_dtype']!='bfloat16'
            or identity['config']['question_order']!=config['question_order']
            or result['updates']!=len(history)):
        raise ValueError('successful BF16 fact-only initializer required')
    for path,digest in config['pins'].items():
        if file_hash(path)!=digest:raise ValueError('initializer source changed')
    adapter_id,adapter_files=checkpoint_identity(initializer/'run/adapter')
    for name,digest in result['adapter_files'].items():
        if file_hash(initializer/'run/adapter'/name)!=digest:raise ValueError('initializer export changed')
    if any(result['adapter_files'].get(name)!=digest for name,digest in adapter_files.items()):
        raise ValueError('initializer identity differs')
    from multimodal_jev.span_training import load_reviewed_bundles
    from multimodal_jev.span_balance import balanced_binary_batches
    bundles,_=load_reviewed_bundles(config['release'],config['manifest_sha256'],config['question_order'])
    _,balance=balanced_binary_batches(bundles,config['seed'])
    if (balance!=read(initializer/'run/balance.json')
            or [h['recordings'] for h in history]!=balance['batch_recording_ids']):
        raise ValueError('initializer fact exposure differs')
    report['initializer']=dict(root=str(initializer),adapter_id=adapter_id,
        base_id=checkpoint_identity(config['model_path'])[0],
        training_recording_ids=sorted({uid for h in history for uid in h['recordings']}),
        files={str(initializer/n):file_hash(initializer/n) for n in
            ('config.json','run/identity.json','run/history.json','run/balance.json','run/result.json','run/terminal.json')},
        adapter_files=adapter_files)
    report.update(learning_release=str(learning),selection_release=str(selection),
        pins={str(root/'MANIFEST.json'):file_hash(root/'MANIFEST.json') for root in (learning,selection)},
        data_audits=dict(learning=parent_audit,selection=selection_audit),
        methods=['sft','joint-grpo','exact-conditional-marginal-ppo'],auxiliary_weights=[0.,.2],
        optimizer=dict(name='BF16MasterAdamW',start='fresh',lr=1e-4,gradient_clip_norm=1.),
        execution_dtype='bfloat16',runtime_profile='qwen3-asr-bf16-hf-order-v1',
        joint_grpo_samples_per_request=32,conditional_integration='exact Boolean support',
        parity_requirement='Independent HF reload and native vLLM full-action gate for every exported version and every next-update request',
        training_arms=['final_only','structured'],
        invalid_combinations=[dict(arm='final_only',auxiliary_weight=.2,reason='no intermediate questions')],
        initializer_requirement='Pinned fact-only SFT; no prior rule-training checkpoint allowed',
        checkpoint_selection='Fixed terminal update, no selection-score picking',
        budget_scope='Same request/update exposure; record tokens, audio duration, rollout and device time separately; not FLOPs matched')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('learning','selection','initializer','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();report=prepare(a.learning,a.selection,a.initializer);atomic_json(a.output,report)
    print(json.dumps({k:report[k] for k in ('heldout_semantic_group','training_semantic_groups','seeds','updates_per_branch')},indent=2))
    print(json.dumps({k:{x:v for x,v in c.items() if x!='request_ids'} for k,c in report['evaluation_cells'].items()},indent=2))
