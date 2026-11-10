from types import SimpleNamespace as NS
import pytest
from multimodal_jev.comparison_schedule import source_batches,comparison_configs,VIEWS,budget_comparison_configs


def examples():
    return [NS(example_id=g+':'+v,split='train',audio=g+'.wav',
               metadata={'source_group':g,'variant':v}) for g in ('a','b','c') for v in VIEWS]


def test_schedule_keeps_complete_groups_and_reproducible_epochs():
    rows=examples();ids=[r.example_id for r in rows]
    batches=source_batches(rows,ids,17,2)
    assert batches==source_batches(reversed(rows),ids,17,2)
    assert len(batches)==6
    for start in (0,3):assert sorted(uid for b in batches[start:start+3] for uid in b)==sorted(ids)
    for batch in batches:
        assert len({uid.split(':')[0] for uid in batch})==1
        assert [uid.split(':')[1] for uid in batch]==list(VIEWS)
    with pytest.raises(ValueError,match='complete'):source_batches(rows,ids[:-1],17,2)
    rows[0].split='valid'
    with pytest.raises(ValueError,match='TRAIN'):source_batches(rows,ids,17,2)


def test_arms_hold_training_settings_fixed_and_do_not_mutate_templates():
    actor=dict(model_path='base',lr=1e-5,lora_rank=4,lora_alpha=8,device='cpu',dtype='float32',cpu_threads=4)
    common=dict(initial_adapter='adapter',initial_checkpoint='merged')
    rl=dict(actor=actor,manifest='train',sampling={'n':2},engine={},**common)
    sft=dict(**actor,train_manifest='train',**common)
    arms=comparison_configs(rl,sft,[['a','b']],29,'exports/29')
    assert len(arms)==5 and rl['sampling']['n']==2 and 'norm_adv_by_std' not in actor
    for name,cfg in arms.items():
        assert cfg['batches']==[['a','b']] and cfg['seed']==29
        if name!='sft':
            assert cfg['sampling']['n']==4 and cfg['engine']['seed']==29
            assert cfg['actor']['norm_adv_by_std']==name.endswith('standard')
    arms['grpo-center']['batches'][0].append('changed')
    assert arms['sft']['batches']==[['a','b']]
    with pytest.raises(ValueError,match='initializer'):
        comparison_configs(rl,{**sft,'initial_adapter':'other'},[['a']],17,'out')
    with pytest.raises(ValueError,match='setting'):
        comparison_configs(rl,{**sft,'lr':1e-4},[['a']],17,'out')


def test_budget_matrix_preserves_schedules_and_isolates_lr_export_paths():
    actor=dict(model_path='base',lr=1e-5,lora_rank=4,lora_alpha=8,device='cpu',dtype='float32',cpu_threads=4)
    common=dict(initial_adapter='adapter',initial_checkpoint='merged')
    rl=dict(actor=actor,manifest='train',sampling={'n':2},engine={},**common)
    sft=dict(**actor,train_manifest='train',**common)
    templates={}
    for seed in (17,29,43):
        templates.update({f'{seed}-{name}.json':cfg for name,cfg in
            comparison_configs(rl,sft,[[f'example-{seed}']],seed,'old').items()})
    result=budget_comparison_configs(templates,'profile','a'*64,'new')
    assert len(result)==45
    exports=[]
    for name,cfg in result.items():
        source=templates[name.split('-lr-')[0]+'.json']
        assert cfg['batches']==source['batches'] and cfg['initial_adapter']==source['initial_adapter']
        assert cfg['budget_control']['limits']=={'training_tokens':765594,'generated_tokens':142848}
        is_sft='-sft-' in name
        current=cfg if is_sft else cfg['actor']
        assert current['lr']==float(name.split('-lr-')[1].removesuffix('.json'))
        assert (source if is_sft else source['actor'])['lr']==1e-5
        if not is_sft:exports.append(cfg['export_root'])
    assert len(exports)==len(set(exports))==36
    with pytest.raises(ValueError,match='fifteen'):
        budget_comparison_configs({},'profile','a'*64,'new')
