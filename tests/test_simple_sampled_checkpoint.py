import pytest
torch=pytest.importorskip('torch')
from multimodal_jev.simple_sampled_checkpoint import reconstruct
from multimodal_jev.span_adapter_export import adapter_key


def fixture():
    names=[f'backbone.language_model.layers.{layer}.self_attn.{proj}.lora_{ab}.default.weight'
           for layer in range(28) for proj in ('q_proj','v_proj') for ab in ('A','B')]
    states=[]
    for rank in range(4):
        group=dict(lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.,amsgrad=False,
            maximize=False,foreach=False,capturable=False,differentiable=False,fused=False,
            decoupled_weight_decay=True,params=list(range(112)))
        states.append(dict(rank=rank,world_size=4,updates=1,stage='sampled_rl_engineering',
            algorithm='grpo_v1',adapter_id='new',reference_policy_id='fixed',names=names,
            parameters={n:torch.tensor([rank+.123456]) for n in names},
            optimizer=dict(param_groups=[group],state={i:dict(step=torch.tensor(1.),
                exp_avg=torch.tensor([.01]),exp_avg_sq=torch.tensor([.001])) for i in range(112)})))
    exported={adapter_key(n,native=True):torch.tensor([r+.123456 for r in range(4)]).reshape(2,2).bfloat16() for n in names}
    return states,exported,dict(algorithm='grpo_v1',adapter_id='new',reference_policy_id='fixed',lr=1e-4,expected_names=names)


def test_actual_shard_order_and_fp32_to_bf16_export():
    states,exported,kw=fixture()
    result=reconstruct(states,exported,**kw)
    assert len(result)==112 and all(torch.equal(result[k],v) for k,v in exported.items())


def test_second_update_requires_continuing_adam_and_explicit_cursor():
    states,exported,kw=fixture()
    for s in states:
        s['updates']=2
        for v in s['optimizer']['state'].values():v['step']=torch.tensor(2.)
    assert len(reconstruct(states,exported,completed_updates=2,**kw))==112
    with pytest.raises(ValueError):reconstruct(states,exported,**kw)
    states[2]['optimizer']['state'][3]['step']=torch.tensor(1.)
    with pytest.raises(ValueError):reconstruct(states,exported,completed_updates=2,**kw)


def test_continuous_stage_is_explicit_and_keeps_exact_adam_cursor():
    states,exported,kw=fixture()
    for s in states:
        s.update(updates=634,stage='continuous_rl_engineering')
        for v in s['optimizer']['state'].values():v['step']=torch.tensor(634.)
    assert len(reconstruct(states,exported,completed_updates=634,
                           expected_stage='continuous_rl_engineering',**kw))==112
    with pytest.raises(ValueError):reconstruct(states,exported,completed_updates=634,**kw)
    with pytest.raises(ValueError):reconstruct(states,exported,completed_updates=635,
                           expected_stage='continuous_rl_engineering',**kw)
    states[0]['optimizer']['state'][0]['step']=torch.tensor(633.)
    with pytest.raises(ValueError):reconstruct(states,exported,completed_updates=634,
                           expected_stage='continuous_rl_engineering',**kw)


def test_formal_stage_requires_explicit_stage_and_full_adam_cursor():
    states, exported, kw = fixture()
    for state in states:
        state.update(updates=634, stage='simple_stage2_rl')
        for value in state['optimizer']['state'].values(): value['step'].fill_(634)
    assert len(reconstruct(states, exported, completed_updates=634, expected_stage='simple_stage2_rl', **kw)) == 112
    with pytest.raises(ValueError):
        reconstruct(states, exported, completed_updates=634, expected_stage='continuous_rl_engineering', **kw)
    states[0]['optimizer']['state'][0]['step'].fill_(1268)
    with pytest.raises(ValueError):
        reconstruct(states, exported, completed_updates=634, expected_stage='simple_stage2_rl', **kw)


@pytest.mark.parametrize('fault',['rank','cursor','reference','algorithm','missing','names','master','dtype','nan','moment','negative','adam_step','lr','export','extra'])
def test_reject_partial_stale_or_corrupt_rl_checkpoint(fault):
    states,exported,kw=fixture();s=states[2];name=kw['expected_names'][0]
    if fault=='rank':states[1],states[2]=states[2],states[1]
    elif fault=='cursor':s['updates']=2
    elif fault=='reference':s['reference_policy_id']='moving'
    elif fault=='algorithm':s['algorithm']='rloo_v1'
    elif fault=='missing':s['optimizer']['state'].pop(111)
    elif fault=='names':s['names']=list(reversed(s['names']))
    elif fault=='master':s['parameters'][name]+=1
    elif fault=='dtype':s['parameters'][name]=s['parameters'][name].bfloat16()
    elif fault=='nan':s['parameters'][name][0]=float('nan')
    elif fault=='moment':s['optimizer']['state'][0]['exp_avg']=torch.ones(2)
    elif fault=='negative':s['optimizer']['state'][0]['exp_avg_sq']*=-1
    elif fault=='adam_step':s['optimizer']['state'][0]['step']=torch.tensor(3.)
    elif fault=='lr':s['optimizer']['param_groups'][0]['lr']=.1
    elif fault=='export':exported[adapter_key(name,native=True)]+=1
    else:exported['unexpected']=torch.zeros(1).bfloat16()
    with pytest.raises(ValueError):reconstruct(states,exported,**kw)
