"""Persisted single-iteration GRPO actor runner, consumed by an external schedule.

Keep the original frozen base and LoRA coordinates across iterations. Sampling
uses merged exports, but optimizer moments belong to the unmerged adapter.
"""
from __future__ import annotations
import json
from pathlib import Path
from .baseline import atomic_json, file_hash
from .policy_identity import checkpoint_identity
from .rollout import Rollout, validate_sampling
from .rl_update import validate_rollouts
from .task_schema import JevExample


def load_rollout_run(run):
    run=Path(run)
    terminal=json.loads((run/'terminal.json').read_text())
    result=json.loads((run/'result.json').read_text())
    identity=json.loads((run/'identity.json').read_text())
    if terminal['exit_code']!=0 or result['status']!='passed':
        raise ValueError('completed successful rollout required')
    config=identity['config'];validate_sampling(config['sampling'])
    policy,files=checkpoint_identity(config['checkpoint'])
    if identity['policy_id']!=policy or result['policy_id']!=policy or identity['checkpoint_files']!=files:
        raise ValueError('rollout checkpoint identity changed')
    if file_hash(config['manifest'])!=identity['manifest_sha256']:
        raise ValueError('rollout manifest changed')
    if file_hash(run/'rollouts.jsonl')!=result['rollouts_sha256']:
        raise ValueError('rollout responses changed')
    selected=config['example_ids']
    if len(selected)!=len(set(selected)):raise ValueError('duplicate selection')
    examples=[]
    with Path(config['manifest']).open() as handle:
        for line in handle:
            row=json.loads(line)
            if row['example_id'] in selected:examples.append(JevExample.from_record(row))
    if len(examples)!=len(selected):raise ValueError('selection mismatch')
    if any(ex.split!='train' for ex in examples):raise ValueError('actor update admits TRAIN only')
    records=[Rollout(**json.loads(line)) for line in (run/'rollouts.jsonl').open()]
    validate_rollouts(examples,records,policy,config['sampling']['n'])
    if result['prompts']!=len(examples) or result['rollouts']!=len(records):
        raise ValueError('rollout count mismatch')
    for ex in examples:
        if file_hash(ex.audio)!=ex.metadata['audio_sha256']:raise ValueError('audio changed')
    return identity,examples,records


def train(config,output):
    import torch
    from peft import LoraConfig, get_peft_model, PeftModel
    from .asr import load_runtime
    from .audio_policy import load_audio
    from .rl_update import update_actor
    from .actor_state import save_optimizer,restore_optimizer
    output=Path(output)
    if (output/'result.json').exists() or (output/'optimizer.pt').exists():
        raise FileExistsError('existing update artifacts; inspect before retry')
    torch.set_num_threads(config['cpu_threads']);torch.manual_seed(config['seed'])
    rollout,examples,records=load_rollout_run(config['rollout_run'])
    parent=Path(config['parent_run']) if config.get('parent_run') else None
    adapter=Path(config['initial_adapter']) if config.get('initial_adapter') else None
    if parent and adapter:raise ValueError('choose parent run or initial adapter')
    base_policy,base_files=checkpoint_identity(config['model_path'])
    optimizer_config={'lr':config['lr'],'weight_decay':0.,'foreach':False,'fused':False}
    prior=None
    if parent:
        terminal=json.loads((parent/'terminal.json').read_text())
        prior=json.loads((parent/'result.json').read_text())
        if terminal['exit_code']!=0 or prior['status']!='passed':raise ValueError('successful parent required')
        if prior['base_policy_id']!=base_policy or prior['optimizer_config']!=optimizer_config:
            raise ValueError('base or optimizer configuration changed')
        if prior['reward_mode']!=config['reward_mode'] or prior['pair_bonus_weight']!=config['pair_bonus_weight']:
            raise ValueError('reward objective changed within trajectory')
        if file_hash(parent/'optimizer.pt')!=prior['optimizer_sha256']:
            raise ValueError('parent optimizer changed')
        adapter=parent/'adapter'
        files={p.name:file_hash(p) for p in sorted(adapter.iterdir()) if p.is_file()}
        if files!=prior['adapter_files']:raise ValueError('parent adapter changed')
        export=json.loads((Path(rollout['config']['checkpoint'])/'EXPORT.json').read_text())
        if export['status']!='passed' or export['adapter_files']!=files or Path(export['training_run']).resolve()!=parent.resolve():
            raise ValueError('rollout is not an export of parent adapter')
    elif adapter:
        files={p.name:file_hash(p) for p in sorted(adapter.iterdir()) if p.is_file()}
        export=json.loads((Path(rollout['config']['checkpoint'])/'EXPORT.json').read_text())
        if export['status']!='passed' or export['adapter_files']!=files or export['base_sha256']!=file_hash(Path(config['model_path'])/'model.safetensors'):
            raise ValueError('initial adapter does not match rollout export')
    elif base_policy!=rollout['policy_id']:
        raise ValueError('unadapted actor must match rollout checkpoint')
    identity={'config':config,'rollout_identity':rollout,'base_policy_id':base_policy,
              'base_files':base_files,'weights_sha256':file_hash(Path(config['model_path'])/'model.safetensors'),
              'manifest_sha256':rollout['manifest_sha256'],'example_ids':[ex.example_id for ex in examples],
              'adapter_files':({p.name:file_hash(p) for p in sorted(adapter.iterdir()) if p.is_file()} if adapter else None),
              'source_sha256':{name:file_hash(Path(__file__).with_name(name)) for name in
                              ('rl_run.py','rl_update.py','verl_audio.py','paired_rollout.py','reward.py','answers.py','audio_policy.py','actor_state.py')}}
    if (output/'identity.json').exists() and json.loads((output/'identity.json').read_text())!=identity:
        raise ValueError('actor retry identity mismatch')
    atomic_json(output/'identity.json',identity)
    processor,base=load_runtime(config['model_path'],config['device'],config['dtype'])
    if adapter:
        model=PeftModel.from_pretrained(base,adapter,is_trainable=True)
    else:
        modules=[name for name,_ in base.named_modules() if name.startswith('model.language_model.')
                 and name.rsplit('.',1)[-1] in {'q_proj','v_proj'}]
        model=get_peft_model(base,LoraConfig(r=config['lora_rank'],lora_alpha=config['lora_alpha'],
                        lora_dropout=0.,target_modules=modules,task_type='CAUSAL_LM',bias='none'))
    parameters=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
    if not parameters:raise ValueError('no trainable adapter parameters')
    optimizer=torch.optim.AdamW([p for _,p in parameters],**optimizer_config)
    if parent:
        restore_optimizer(model,optimizer,parent/'optimizer.pt',config['device'])
    metrics=update_actor(model,processor,optimizer,examples,[load_audio(ex.audio) for ex in examples],records,
                         rollout['policy_id'],rollout['config']['sampling']['n'],
                         max_logprob_error=config['max_logprob_error'],reward_mode=config['reward_mode'],
                         pair_bonus_weight=config['pair_bonus_weight'],
                         micro_batch_size=config.get('micro_batch_size'))
    model.save_pretrained(output/'adapter');processor.save_pretrained(output/'processor')
    save_optimizer(model,optimizer,output/'optimizer.pt')
    # Keep the exporter's base/train fields while retaining the complete run config.
    export_config={**config,'train_manifest':rollout['config']['manifest']}
    result={'status':'passed','config':export_config,'base_policy_id':base_policy,
            'optimizer_config':optimizer_config,'optimizer_sha256':file_hash(output/'optimizer.pt'),
            'adapter_files':{p.name:file_hash(p) for p in sorted((output/'adapter').iterdir()) if p.is_file()},
            'updates':(prior['updates'] if prior else 0)+int(metrics['updated']),
            'iteration':(prior['iteration'] if prior else 0)+1,
            'reward_mode':config['reward_mode'],'pair_bonus_weight':config['pair_bonus_weight'],
            'metrics':metrics,'scope':'one persisted actor iteration; export and next fresh rollout required; no efficacy claim'}
    atomic_json(output/'result.json',result)
    return result
