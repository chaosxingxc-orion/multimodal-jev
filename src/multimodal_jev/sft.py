"""Resumable audio-schema LoRA SFT initialization before RL comparisons."""
from __future__ import annotations
import hashlib,json,random,time
from pathlib import Path
from .baseline import atomic_json,file_hash,select_examples


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


def train(config,output):
    import torch
    from peft import LoraConfig,get_peft_model,get_peft_model_state_dict,set_peft_model_state_dict
    from .asr import load_runtime
    from .audio_policy import teacher_forcing,load_audio
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'result.json').exists():raise FileExistsError('SFT completed')
    torch.set_num_threads(config['cpu_threads']);torch.manual_seed(config['seed'])
    rows=select_training(config)
    identity={'config':config,'manifest_sha256':file_hash(config['train_manifest']),
              'weights_sha256':file_hash(Path(config['model_path'])/'model.safetensors'),
              'code_sha256':{n:file_hash(Path(__file__).with_name(n)) for n in ('sft.py','audio_policy.py','asr.py','baseline.py')},
              'example_ids':[ex.example_id for ex in rows]}
    if (output/'identity.json').exists() and json.loads((output/'identity.json').read_text())!=identity:
        raise ValueError('SFT resume identity mismatch')
    atomic_json(output/'identity.json',identity)
    processor,model=load_runtime(config['model_path'],config['device'],config['dtype'])
    modules=[name for name,_ in model.named_modules() if name.startswith('model.language_model.') and name.rsplit('.',1)[-1] in {'q_proj','v_proj'}]
    lora=LoraConfig(r=config['lora_rank'],lora_alpha=config['lora_alpha'],lora_dropout=0.0,
                    target_modules=modules,task_type='CAUSAL_LM',bias='none')
    model=get_peft_model(model,lora)
    # Eval disables all dropout while preserving autograd; same behavior as RL actor.
    model.eval()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr'],
                                weight_decay=0.0,foreach=False,fused=False)
    checkpoint=output/'resume.pt';step=0;history=[]
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        set_peft_model_state_dict(model,saved['adapter']);optimizer.load_state_dict(saved['optimizer'])
        step=saved['step'];history=saved['history'];torch.set_rng_state(saved['rng'])
    start=time.monotonic();total=(len(rows)+config['accumulate']-1)//config['accumulate']
    for index in range(step,total):
        subset=rows[index*config['accumulate']:(index+1)*config['accumulate']]
        optimizer.zero_grad(set_to_none=True);losses=[]
        for ex in subset:
            if file_hash(ex.audio)!=ex.metadata['audio_sha256']:raise ValueError('source audio changed')
            inputs,_=teacher_forcing(processor,ex,load_audio(ex.audio))
            inputs={k:v.to(device=model.device,dtype=model.dtype if v.is_floating_point() else v.dtype) for k,v in inputs.items()}
            loss=model(**inputs,use_cache=False).loss
            if not torch.isfinite(loss):raise ValueError('nonfinite SFT loss')
            (loss/len(subset)).backward();losses.append(float(loss.detach()))
            atomic_json(output/'progress.json',{'step':index,'total_steps':total,'phase':'backward','examples_in_step':len(losses),'heartbeat_unix':time.time()})
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        optimizer.step()
        history.append({'step':index+1,'loss':sum(losses)/len(losses),'gradient_norm':float(norm)})
        state={'step':index+1,'adapter':get_peft_model_state_dict(model),'optimizer':optimizer.state_dict(),
               'history':history,'rng':torch.get_rng_state()}
        temp=output/'resume.tmp';torch.save(state,temp);temp.replace(checkpoint)
        atomic_json(output/'progress.json',{'step':index+1,'total_steps':total,'loss':history[-1]['loss'],'heartbeat_unix':time.time(),'elapsed_this_session_seconds':time.monotonic()-start})
    model.save_pretrained(output/'adapter');processor.save_pretrained(output/'processor')
    result={'status':'passed','examples':len(rows),'source_groups':len({ex.metadata['source_group'] for ex in rows}),
            'updates':total,'history':history,'config':config,
            'scope':'SFT initialization development run; adapter not yet evaluated or exported to vLLM',
            'elapsed_this_session_seconds':time.monotonic()-start}
    atomic_json(output/'result.json',result)
    return result
