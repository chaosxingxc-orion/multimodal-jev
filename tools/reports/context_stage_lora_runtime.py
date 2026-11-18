"""PEFT stage transition candidate; production rollout admission is separate.

Operate before FSDP/runtime patching. Preserve the frozen base and Stage1
adapter without merging. New Stage2 adapters start with zero residual.
"""
from multimodal_jev.context_stage_lora import transition,from_policy


def configure_stage2(adapted,stage1_contract,stage1,stage2,*,reuse_stage1_adapter):
    import torch
    from peft import LoraConfig
    from multimodal_jev.context_policy_snapshot import lora_contract
    plan=transition(stage1,stage2,reuse_stage1_adapter=reuse_stage1_adapter)
    if from_policy(stage1_contract)!=plan['stage1']:
        raise ValueError('Stage1 policy does not match configured adapter')
    backbone=adapted.get_base_model().model
    if lora_contract(backbone)!=stage1_contract['lora'] or set(adapted.peft_config)!={'default'}:
        raise ValueError('one authenticated active Stage1 adapter required before transition')
    if plan['mode']=='reuse':
        adapted.set_adapter('default')
        return dict(plan=plan,adapter_names={'stage1':'default','stage2':'default'},production_admitted=False)
    # Keep the existing PEFT name to avoid altering Stage1 weight identity.
    target=plan['stage2'];available=dict(backbone.named_modules())
    if any(name not in available for name in target['target_modules']):
        raise ValueError('all configured Stage2 target modules must exist')
    config=LoraConfig(r=target['rank'],lora_alpha=target['alpha'],lora_dropout=0.,bias='none',
        target_modules=['model.'+name for name in target['target_modules']],task_type='CAUSAL_LM',
        init_lora_weights=True,use_rslora=False,use_dora=False)
    # PEFT's model-level set_adapter accepts one name; the tuner supports additive adapters.
    adapted.add_adapter('stage2',config)
    adapted.base_model.set_adapter(['default','stage2'])
    for name,p in adapted.named_parameters():
        if '.lora_A.stage2.' in name or '.lora_B.stage2.' in name:
            p.data=p.data.to(torch.bfloat16);p.requires_grad_(True)
        else:p.requires_grad_(False)
    for module in backbone.modules():
        if hasattr(module,'lora_B') and 'stage2' in module.lora_B:
            if torch.count_nonzero(module.lora_B['stage2'].weight).item()!=0:
                raise ValueError('Stage2 must start with zero residual')
    return dict(plan=plan,adapter_names={'stage1':'default','stage2':'stage2'},production_admitted=False)
