"""Stage-specific LoRA identity and transition contract.

Equality refers to execution configuration, not stage labels or optimizer LR.
Different adapters preserve Stage1 as a frozen residual; never merge the base.
The runtime must explicitly admit this composition before dispatching it.
"""
import hashlib
import json
import math
import re


FIELDS={'rank','alpha','target_modules','dropout','bias','use_rslora','use_dora',
        'fan_in_fan_out','execution_dtype'}


def normalize(config):
    if not isinstance(config,dict) or set(config)!=FIELDS:
        raise ValueError('complete explicit stage LoRA execution configuration required')
    c=dict(config)
    if (type(c['rank']) is not int or c['rank']<1
            or type(c['alpha']) not in (int,float) or not math.isfinite(c['alpha']) or c['alpha']<=0
            or c['dropout']!=0 or c['bias']!='none' or c['use_rslora'] is not False
            or c['use_dora'] is not False or c['fan_in_fan_out'] is not False
            or c['execution_dtype']!='bfloat16'):
        raise ValueError('ordinary BF16 bias-free zero-dropout stage LoRA required')
    targets=c['target_modules']
    if (not isinstance(targets,list) or not targets or len(set(targets))!=len(targets)
            or any(not isinstance(t,str) or not re.fullmatch(r'language_model\.layers\.\d+\.self_attn\.[qv]_proj',t)
                   for t in targets)):
        raise ValueError('resolved unique language q/v target modules required')
    c['target_modules']=sorted(targets);c['alpha']=float(c['alpha']);c['dropout']=0.
    return c


def identity(config):
    return hashlib.sha256(json.dumps(normalize(config),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def transition(stage1,stage2,*,reuse_stage1_adapter):
    first,second=normalize(stage1),normalize(stage2)
    if type(reuse_stage1_adapter) is not bool:
        raise ValueError("explicit Boolean reuse_stage1_adapter required")
    if reuse_stage1_adapter and first!=second:
        raise ValueError("cannot reuse Stage1 adapter with a different execution configuration")
    reuse=reuse_stage1_adapter
    return dict(schema='jev-stage-lora-transition-v1',stage1=first,stage2=second,
        reuse_stage1_adapter=reuse,configurations_equal=first==second,
        stage1_config_id=identity(first),stage2_config_id=identity(second),
        mode='reuse' if reuse else 'frozen_stage1_plus_stage2',
        active_adapters=['default'] if reuse else ['stage1','stage2'],
        trainable_adapters=['default'] if reuse else ['stage2'],
        frozen_adapters=[] if reuse else ['stage1'],
        stage2_initial_function='stage1_endpoint',stage2_optimizer='fresh',
        stage2_new_adapter_initialization=None if reuse else 'random_A_zero_B',
        reference='immutable_stage1_endpoint_outputs',base_merged=False)


def from_policy(contract):
    specs=contract['lora']
    settings={(s['rank'],s['alpha']) for s in specs.values()}
    if len(settings)!=1:raise ValueError('uniform policy LoRA required')
    rank,alpha=next(iter(settings))
    for s in specs.values():
        if (s['scaling']!=alpha/rank or s['active_adapter']!='default'
                or s['dropout']!=0 or s['use_dora'] is not False or s['fan_in_fan_out'] is not False):
            raise ValueError('policy LoRA execution semantics differ')
    return normalize(dict(rank=rank,alpha=alpha,target_modules=list(specs),dropout=0.,bias='none',
        use_rslora=False,use_dora=False,fan_in_fan_out=False,execution_dtype='bfloat16'))


def bind_design(design,source_contract):
    plan=transition(design['stage1']['lora'],design['stage2']['lora'],
        reuse_stage1_adapter=design['stage2']['reuse_stage1_adapter'])
    if plan['stage1']!=from_policy(source_contract):
        raise ValueError('Stage1 LoRA differs from authenticated initialization/runtime')
    return plan


def admit_stage2(plan,source_contract,*,stacked_runtime_admitted=False):
    expected=transition(plan['stage1'],plan['stage2'],reuse_stage1_adapter=plan['reuse_stage1_adapter'])
    if plan!=expected or expected['stage1']!=from_policy(source_contract):
        raise ValueError('stage transition or Stage1 checkpoint identity differs')
    if expected['mode']!='reuse' and not stacked_runtime_admitted:
        raise ValueError('distinct stage adapters require admitted stacked actor/export/vLLM runtime')
    return expected
