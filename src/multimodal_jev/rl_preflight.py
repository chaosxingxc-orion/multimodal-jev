"""Offline validation of an audio RL schedule before allocating an engine.

This checks artifacts and TRAIN selection; it does not execute vLLM or prove
model quality, numerical parity, accelerator availability, or an optimizer step.
"""
import json
import math
from pathlib import Path
from .baseline import file_hash
from .pair_reward import paired_rewards
from .policy_identity import checkpoint_identity,validate_adapter_export
from .rollout import validate_sampling
from .task_schema import JevExample


def validate_schedule(config):
    actor=config.get('actor')
    if not isinstance(actor,dict) or not actor.get('model_path'):
        raise ValueError('actor.model_path must be explicit and point to the frozen LoRA base')
    base,checkpoint=Path(actor['model_path']),Path(config['initial_checkpoint'])
    for name,path in [('actor base',base),('initial rollout',checkpoint)]:
        if not path.is_dir() or not (path/'config.json').is_file():
            raise ValueError(f'{name} checkpoint is unavailable: {path}')
    validate_sampling(config['sampling'])
    if type(config.get('seed')) is not int or config['seed']<0:
        raise ValueError('nonnegative integer schedule seed required')
    if config.get('reward_mode') not in {'grpo','pair-grpo'}:
        raise ValueError('unsupported reward mode')
    bonus=config.get('pair_bonus_weight')
    if type(bonus) not in (int,float) or not math.isfinite(bonus) or not 0<=bonus<=1:
        raise ValueError('pair_bonus_weight must be in [0,1]')
    batches=config.get('batches')
    if not isinstance(batches,list) or not batches:
        raise ValueError('explicit TRAIN batch schedule required')
    for batch in batches:
        if (not isinstance(batch,list) or not batch or
                any(not isinstance(uid,str) or not uid for uid in batch)):
            raise ValueError('each batch must contain nonempty example IDs')
        if len(batch)!=len(set(batch)):raise ValueError('duplicate selection within batch')
    base_policy,base_files=checkpoint_identity(base)
    initial_policy,initial_files=((base_policy,base_files) if base.resolve()==checkpoint.resolve()
                                  else checkpoint_identity(checkpoint))
    adapter_files=None
    if config.get('initial_adapter'):
        adapter_files=validate_adapter_export(base,checkpoint,config['initial_adapter'],
                                              base_files=base_files,checkpoint_files=initial_files)
    elif base_policy!=initial_policy:
        raise ValueError('different actor base requires an explicit initial_adapter')
    wanted={uid for batch in batches for uid in batch};selected={}
    manifest=Path(config['manifest']);manifest_hash=file_hash(manifest)
    with manifest.open() as handle:
        for line in handle:
            row=json.loads(line);uid=row['example_id']
            if uid not in wanted:continue
            if uid in selected:raise ValueError(f'duplicate manifest example: {uid}')
            example=JevExample.from_record(row)
            if example.split!='train':raise ValueError('schedule admits TRAIN only')
            selected[uid]=example
    if selected.keys()!=wanted:raise ValueError('schedule selection missing from manifest')
    audio_hashes={}
    for example in selected.values():
        if example.audio not in audio_hashes:audio_hashes[example.audio]=file_hash(example.audio)
        if audio_hashes[example.audio]!=example.metadata.get('audio_sha256'):
            raise ValueError(f'audio changed: {example.example_id}')
    if config['reward_mode']=='pair-grpo':
        for batch in batches:
            examples=[selected[uid] for uid in batch]
            # Exercise only the relation contract, before any sampling. The
            # gold values here never become rollout inputs or training responses.
            paired_rewards(examples,[ex.target for ex in examples],bonus)
    return {'initial_policy_id':initial_policy,'actor_base_policy_id':base_policy,
            'initial_adapter_files':adapter_files,'manifest_sha256':manifest_hash,
            'audio_sha256':audio_hashes,'iterations':len(batches),
            'unique_examples':len(selected),'unique_audio':len(audio_hashes),
            'scheduled_prompts':sum(map(len,batches)),
            'scheduled_responses':sum(map(len,batches))*config['sampling']['n']}
