import json
from pathlib import Path

import pytest

from tools.reports.compile_simple_experiment import ROOT, compile_recipe


def recipe():
    return json.loads((ROOT/'configs/experiments/simple-two-stage-recipe.json').read_bytes())


@pytest.mark.parametrize('cards,rank,end,batch',[(8,64,160,384),(4,16,318,128)])
def test_changed_recipe_carries_to_schedule_dataset_and_adapter(cards,rank,end,batch):
    cfg=recipe()
    cfg.update(total_devices=cards,stage1_lora=dict(rank=rank,alpha=rank*2),
               stage2_lora=dict(rank=rank,alpha=rank*2))
    files=compile_recipe(cfg,Path('configs/experiments/example'))
    d=json.loads(files['two-stage.json'])
    assert d['stage1']['evaluation_steps'][-1]==d['stage1']['reference_endpoint']==end
    assert d['stage2']['lora']['rank']==d['stage1']['lora']['rank']==rank
    assert d['stage2']['macro_queries']==batch
    assert d['stage2']['batch_config']['source_queries'] is None
    assert json.loads(files['dataset.json'])['batch_config']['training_devices']==cards-cfg['rollout_instances']


def test_mismatched_reuse_fails_before_publication():
    c=recipe();c['stage2_lora']['rank']=16
    with pytest.raises(ValueError,match='cannot reuse'):
        compile_recipe(c,Path('configs/experiments/example'))
