import copy

import pytest

from multimodal_jev.context_stage_lora import normalize,identity,transition,bind_design,admit_stage2


def config():
    return dict(rank=4,alpha=8,target_modules=['language_model.layers.0.self_attn.q_proj',
        'language_model.layers.0.self_attn.v_proj'],dropout=0.,bias='none',use_rslora=False,
        use_dora=False,fan_in_fan_out=False,execution_dtype='bfloat16')


def contract(c):
    return {'lora':{name:dict(rank=c['rank'],alpha=c['alpha'],scaling=c['alpha']/c['rank'],
        active_adapter='default',dropout=0.,use_dora=False,fan_in_fan_out=False)
        for name in c['target_modules']}}


def test_same_execution_configuration_reuses_one_adapter_without_mutating_inputs():
    a=config();b=copy.deepcopy(a);b['target_modules'].reverse();b['alpha']=8.
    original=copy.deepcopy(b);plan=transition(a,b,reuse_stage1_adapter=True)
    assert identity(a)==identity(b) and plan['mode']=='reuse'
    assert plan['active_adapters']==plan['trainable_adapters']==['default']
    assert not plan['frozen_adapters'] and b==original
    assert admit_stage2(plan,contract(a))==plan


@pytest.mark.parametrize('change',[{'rank':8},{'alpha':16},{'target_modules':['language_model.layers.0.self_attn.q_proj']}])
def test_different_configs_preserve_stage1_and_require_separate_runtime_admission(change):
    a=config();b=dict(a,**change);plan=transition(a,b,reuse_stage1_adapter=False)
    assert plan['mode']=='frozen_stage1_plus_stage2'
    assert plan['active_adapters']==['stage1','stage2'] and plan['trainable_adapters']==['stage2']
    assert plan['frozen_adapters']==['stage1'] and plan['base_merged'] is False
    assert plan['stage2_new_adapter_initialization']=='random_A_zero_B'
    with pytest.raises(ValueError,match='admitted stacked'):admit_stage2(plan,contract(a))


@pytest.mark.parametrize('change',[{'dropout':.1},{'use_dora':True},{'rank':True},{'alpha':float('nan')},
    {'execution_dtype':'float32'},{'target_modules':['q_proj']},{'use_rslora':True}])
def test_unsupported_or_ambiguous_config_rejected(change):
    with pytest.raises(ValueError):normalize(dict(config(),**change))


def test_binding_rejects_wrong_stage1_and_forged_reuse():
    a=config();design=dict(stage1={'lora':a},stage2={'lora':dict(a,rank=8),'reuse_stage1_adapter':False})
    plan=bind_design(design,contract(a))
    with pytest.raises(ValueError):bind_design(design,contract(dict(a,rank=2)))
    plan['mode']='reuse'
    with pytest.raises(ValueError):admit_stage2(plan,contract(a))


def test_formal_stage1_binds_config_to_actual_adapter_and_stage2_resume_contract():
    import json
    from pathlib import Path
    from tools.reports.context_infonce_formal import formal_config
    from tools.reports.context_cached_reference_trainer import generate
    root=Path(__file__).resolve().parents[1]
    design=json.loads((root/'configs/experiments/context-two-stage-v0.0.8.json').read_text())
    original=dict(design['stage1'],physical_devices=list(range(8)),microbatch=8,
        source_contract=contract(design['stage1']['lora']),
        source_root='/old/source',schedule='/old/source/schedule',parent_schedule='/old/source/parent-schedule',pins={})
    c=formal_config(original,'/old','/new',design)
    assert c['stage_lora_transition']['mode']=='reuse'
    for formal in (False,True):
        source=generate(root,formal=formal)
        assert "contract['stage_lora_transition']=admit_stage2(c['stage_lora_transition'],c['source_contract'])" in source
        compile(source,'<stage-lora>','exec')


def test_same_config_can_explicitly_request_fresh_parameters():
    a=config();plan=transition(a,a,reuse_stage1_adapter=False)
    assert plan['configurations_equal'] is True and plan['reuse_stage1_adapter'] is False
    assert plan['mode']=='frozen_stage1_plus_stage2'
    assert plan['trainable_adapters']==['stage2'] and plan['frozen_adapters']==['stage1']
    with pytest.raises(ValueError,match='admitted stacked'):admit_stage2(plan,contract(a))


@pytest.mark.parametrize('flag',[None,0,1,'true','false'])
def test_reuse_flag_is_strict_boolean(flag):
    with pytest.raises(ValueError,match='Boolean'):transition(config(),config(),reuse_stage1_adapter=flag)


def test_incompatible_config_cannot_request_reuse():
    with pytest.raises(ValueError,match='cannot reuse'):
        transition(config(),dict(config(),rank=8),reuse_stage1_adapter=True)
