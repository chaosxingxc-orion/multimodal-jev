"""Resumable audio-schema LoRA SFT initialization before RL comparisons."""
from __future__ import annotations
import hashlib,json,random,time
from pathlib import Path
from .baseline import atomic_json,file_hash,select_examples
from .training_budget import batch_budget,sum_budgets


def initialization_identity(config):
    """Require the same original-base/adapter/export lineage as the RL arm."""
    from .policy_identity import checkpoint_identity,validate_adapter_export
    adapter=config.get('initial_adapter');checkpoint=config.get('initial_checkpoint')
    if bool(adapter)!=bool(checkpoint):
        raise ValueError('SFT continuation requires initial_adapter and initial_checkpoint together')
    if not adapter:return None
    base_policy,base_files=checkpoint_identity(config['model_path'])
    policy,files=checkpoint_identity(checkpoint)
    adapter_files=validate_adapter_export(config['model_path'],checkpoint,adapter,
                                         base_files=base_files,checkpoint_files=files)
    saved=json.loads((Path(adapter)/'adapter_config.json').read_text())
    if saved.get('r')!=config['lora_rank'] or saved.get('lora_alpha')!=config['lora_alpha']:
        raise ValueError('SFT continuation LoRA rank/alpha differs from saved adapter')
    return {'base_policy_id':base_policy,'base_files':base_files,'policy_id':policy,
            'checkpoint_files':files,'adapter_files':adapter_files,
            'optimizer_start':'fresh AdamW; initializer optimizer is not restored'}


def initialize_adapter(base,config):
    from peft import LoraConfig,get_peft_model,PeftModel
    if config.get('initial_adapter'):
        return PeftModel.from_pretrained(base,config['initial_adapter'],is_trainable=True)
    modules=[name for name,_ in base.named_modules() if name.startswith('model.language_model.')
             and name.rsplit('.',1)[-1] in {'q_proj','v_proj'}]
    return get_peft_model(base,LoraConfig(r=config['lora_rank'],lora_alpha=config['lora_alpha'],
                         lora_dropout=0.0,target_modules=modules,task_type='CAUSAL_LM',bias='none'))


def select_training(config):
    from .task_schema import JevExample
    native=select_examples(config['train_manifest'],'train',config['groups_per_class'])
    groups={ex.metadata['source_group'] for ex in native}
    rows=[]
    with Path(config['train_manifest']).open() as f:
        for line in f:
            record=json.loads(line)
            if record['metadata']['source_group'] in groups:
                rows.append(JevExample.from_record(record))
    expected={'native','permutation','rebind','missing_candidate','noul.True','noul.False'}
    for group in groups:
        variants=[ex.metadata['variant'] for ex in rows if ex.metadata['source_group']==group]
        if set(variants)!=expected or len(variants)!=len(expected):raise ValueError('incomplete SFT source group')
    if any(ex.split!='train' for ex in rows):raise ValueError('SFT admits train only')
    random.Random(config['seed']).shuffle(rows)
    return rows


def training_batches(config):
    """Explicit update boundaries for comparisons, or legacy initialization."""
    from .task_schema import JevExample
    if 'batches' not in config:
        size=config.get('accumulate')
        if type(size) is not int or size<1:raise ValueError('positive integer accumulate required')
        rows=select_training(config)
        return [rows[i:i+size] for i in range(0,len(rows),size)]
    if 'accumulate' in config or 'groups_per_class' in config:
        raise ValueError('explicit batches cannot also set accumulate or groups_per_class')
    schedule=config['batches']
    if not isinstance(schedule,list) or not schedule:raise ValueError('nonempty batch schedule required')
    for ids in schedule:
        if not isinstance(ids,list) or not ids or any(not isinstance(uid,str) or not uid for uid in ids):
            raise ValueError('nonempty example IDs required in each batch')
        if len(ids)!=len(set(ids)):raise ValueError('duplicate ID within SFT batch')
    wanted={uid for ids in schedule for uid in ids};selected={}
    with Path(config['train_manifest']).open() as handle:
        for line in handle:
            row=json.loads(line);uid=row['example_id']
            if uid not in wanted:continue
            if uid in selected:raise ValueError('duplicate manifest ID')
            ex=JevExample.from_record(row)
            if ex.split!='train':raise ValueError('SFT admits train only')
            selected[uid]=ex
    if set(selected)!=wanted:raise ValueError('SFT selection missing from manifest')
    # Reuse across different updates is intentional and retained in the ledger.
    return [[selected[uid] for uid in ids] for ids in schedule]


def train(config,output):
    import torch
    from peft import get_peft_model_state_dict,set_peft_model_state_dict
    from .asr import load_runtime
    from .audio_policy import teacher_forcing,load_audio
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'result.json').exists():raise FileExistsError('SFT completed')
    initialization=initialization_identity(config)
    torch.set_num_threads(config['cpu_threads']);torch.manual_seed(config['seed'])
    batches=training_batches(config);rows=[ex for batch in batches for ex in batch]
    identity={'config':config,'manifest_sha256':file_hash(config['train_manifest']),
              'weights_sha256':file_hash(Path(config['model_path'])/'model.safetensors'),
              'code_sha256':{n:file_hash(Path(__file__).with_name(n)) for n in ('sft.py','audio_policy.py','asr.py','baseline.py','training_budget.py','policy_identity.py')},
              'example_ids':[ex.example_id for ex in rows],
              'batch_example_ids':[[ex.example_id for ex in batch] for batch in batches]}
    if initialization is not None:
        identity['initialization']=initialization
        identity['base_files']=initialization['base_files']
    if (output/'identity.json').exists() and json.loads((output/'identity.json').read_text())!=identity:
        raise ValueError('SFT resume identity mismatch')
    atomic_json(output/'identity.json',identity)
    processor,model=load_runtime(config['model_path'],config['device'],config['dtype'])
    model=initialize_adapter(model,config)
    # Eval disables all dropout while preserving autograd; same behavior as RL actor.
    model.eval()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr'],
                                weight_decay=0.0,foreach=False,fused=False)
    checkpoint=output/'resume.pt';step=0;history=[]
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        set_peft_model_state_dict(model,saved['adapter']);optimizer.load_state_dict(saved['optimizer'])
        step=saved['step'];history=saved['history'];torch.set_rng_state(saved['rng'])
    start=time.monotonic();total=len(batches)
    for index in range(step,total):
        subset=batches[index]
        optimizer.zero_grad(set_to_none=True);losses=[];budgets=[]
        for ex in subset:
            if file_hash(ex.audio)!=ex.metadata['audio_sha256']:raise ValueError('source audio changed')
            inputs,boundary=teacher_forcing(processor,ex,load_audio(ex.audio))
            budgets.append(batch_budget([boundary],[int((inputs['labels']!=-100).sum())],
                           [int(inputs['input_features_mask'].sum())],gradient_passes=1,
                           no_grad_passes=0,optimizer_steps=0))
            inputs={k:v.to(device=model.device,dtype=model.dtype if v.is_floating_point() else v.dtype) for k,v in inputs.items()}
            loss=model(**inputs,use_cache=False).loss
            if not torch.isfinite(loss):raise ValueError('nonfinite SFT loss')
            (loss/len(subset)).backward();losses.append(float(loss.detach()))
            atomic_json(output/'progress.json',{'step':index,'total_steps':total,'phase':'backward','examples_in_step':len(losses),'heartbeat_unix':time.time()})
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        optimizer.step()
        budget=sum_budgets(budgets);budget['optimizer_steps']=1
        history.append({'step':index+1,'loss':sum(losses)/len(losses),'gradient_norm':float(norm),
                        'budget':budget})
        state={'step':index+1,'adapter':get_peft_model_state_dict(model),'optimizer':optimizer.state_dict(),
               'history':history,'rng':torch.get_rng_state()}
        temp=output/'resume.tmp';torch.save(state,temp);temp.replace(checkpoint)
        atomic_json(output/'progress.json',{'step':index+1,'total_steps':total,'loss':history[-1]['loss'],'heartbeat_unix':time.time(),'elapsed_this_session_seconds':time.monotonic()-start})
    model.save_pretrained(output/'adapter');processor.save_pretrained(output/'processor')
    for path in (output/'adapter').iterdir():
        if path.is_file():path.chmod(path.stat().st_mode | 0o444)
    result={'status':'passed','examples':len(rows),'source_groups':len({ex.metadata['source_group'] for ex in rows}),
            'updates':total,'history':history,'config':config,
            'training_budget':sum_budgets(row['budget'] for row in history),
            'initialization':initialization,
            'adapter_files':{p.name:file_hash(p) for p in sorted((output/'adapter').iterdir()) if p.is_file()},
            'scope':('SFT continuation from shared initializer; not yet evaluated; matched-budget comparison requires a separately fixed schedule'
                     if initialization else 'SFT initialization development run; adapter not yet evaluated or exported to vLLM'),
            'elapsed_this_session_seconds':time.monotonic()-start}
    atomic_json(output/'result.json',result)
    return result
