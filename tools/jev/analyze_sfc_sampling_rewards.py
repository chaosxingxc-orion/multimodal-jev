#!/usr/bin/env python3
"""Learning-only reward signal after independent complete TRAIN78 sampling acceptance."""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.omni_sampling_acceptance import audit_records,read_json
from multimodal_jev.sfc_reward_batch import compose,digest
from multimodal_jev.sfc_reward_signal import summarize
from sample_sfc_omni_learning import load_inputs


def analyze(root,acceptance,acceptance_hash):
    if file_hash(acceptance)!=acceptance_hash:raise ValueError('registered sampling acceptance changed')
    accepted=read_json(acceptance);launch=read_json(root/'launch.json')
    run=root/'run';result=read_json(run/'result.json');identity=read_json(run/'identity.json')
    if (accepted['container_id']!=launch['container_id']
            or accepted['source_commit']!=launch['source_commit']
            or accepted['result_sha256']!=file_hash(run/'result.json')
            or accepted['records_sha256']!=file_hash(run/'records.jsonl')
            or accepted['weight_updates']!=0 or accepted['responses']!=156
            or result['identity_sha256']!=file_hash(run/'identity.json')):
        raise ValueError('complete immutable sampling acceptance required')
    config=read_json(root/'config.json');base,rows,preflight,checks=load_inputs(config)
    if identity['config']!=config or identity['inputs_sha256']!=digest(rows):
        raise ValueError('accepted input identity changed')
    for name,pin in preflight['processor_assets'].items():
        if file_hash(Path(base['audio_model'])/name)!=pin:raise ValueError('fixed tokenizer assets required')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(base['audio_model'],local_files_only=True)
    audited=audit_records(run,rows,checks,tokenizer,policy_sha256=identity['policy_sha256'],
        instruction_sha256=identity['instruction_sha256'],sampling=identity['effective_sampling'])
    if any(accepted.get(k)!=v for k,v in audited.items()):raise ValueError('independent raw evidence changed')
    # All sampling identities are checked before opening any targets. Only the
    # existing TRAIN release is opened; selection rows are not passed to rewards.
    release=Path(preflight['release']);manifest=read_json(release/'MANIFEST.json')
    target_path=release/'train_id.targets.jsonl'
    if file_hash(target_path)!=manifest['splits']['train_id']['targets_sha256']:
        raise ValueError('registered TRAIN targets changed')
    selected={r['id'] for r in rows}
    targets=[r for line in target_path.open() if (r:=json.loads(line))['id'] in selected]
    all_rows=[json.loads(line) for line in (release/'train_id.requests.jsonl').open()]
    partition=read_json(config['partition']);records=[json.loads(line) for line in (run/'records.jsonl').open()]
    evidence={key:{k:checks[key][k] for k in ('prompt_sha256','input_ids_sha256')} for key in checks}
    reports={}
    for mode in ('independent','function-conjunction'):
        batch=compose(all_rows,partition,rows,targets,records,evidence,
            policy_sha256=identity['policy_sha256'],instruction_sha256=identity['instruction_sha256'],
            group_size=2,tokenizer=tokenizer,mode=mode,bonus_weight=.2)
        reports[mode]=summarize(records,batch)
    return dict(schema='sfc-accepted-sampling-reward-signal-v1',acceptance_sha256=acceptance_hash,
        records_sha256=file_hash(run/'records.jsonl'),selected_targets_sha256=digest(targets),
        train_targets_file_sha256=file_hash(target_path),selected_target_bundles=len(targets),
        partition_sha256=config['partition_sha256'],reports=reports,weight_updates=0,
        scope='Descriptive rewards on complete independently accepted learning TRAIN78 K2 samples; '
              'whole TRAIN target file parsed to select six learning bundles, no TEST targets, '
              'no reward tuning, actor gradient or efficacy claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('root','acceptance','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--acceptance-hash',required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve prior reward report')
    atomic_json(a.output,analyze(a.root,a.acceptance,a.acceptance_hash))
