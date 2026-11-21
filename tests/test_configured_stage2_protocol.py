import copy
import json
from pathlib import Path
import pytest
from tools.reports import compile_simple_experiment as compiler
from tools.reports.context_dual_actor_candidate import options_check
from multimodal_jev.stage2_view_runtime import actor_contract,editable_objective


def test_compile_to_actor_and_relocated_checkpoint_identity():
    recipe=json.loads((compiler.ROOT/'configs/experiments/simple-two-stage-recipe.json').read_bytes())
    recipe.update(total_devices=5,rollout_instances=2,advantage_std_floor=.55,rollout_batch=48)
    files=compiler.compile_recipe(recipe,Path('configs/experiments/alternate'))
    design=json.loads(files['two-stage.json']);spec=json.loads(files['stage2-runtime.json']);task=design['stage2']
    assert task['macro_queries']==192 and spec['serving']['max_batch_queries']==48
    assert spec['panel_config']['minimum_queries']>=task['macro_queries']
    ref=compiler.load(recipe['objective'])
    execution=compiler.load(dict(path=ref['execution']['source'],sha256=ref['execution']['sha256']))
    execution['actor_seed']=811
    numeric=dict(advantage_std_floor=.55,policy_clip=execution['policy_clip'])
    cfg=dict(algorithm=task['algorithm'],dual_arm=task['arm'],source_policy_id='endpoint',seed=1,
        objective_config=recipe['objective'],view_lineage={'objective_design_sha256':recipe['objective']['sha256']},
        objective_plugin=dict(path='/first/plugin',sha256=ref['objective_sha256'],options=ref['options'],numerics=numeric),
        view_execution=execution,stage2_objective={k:task[k] for k in ('algorithm','arm','group_size','clip_ratio','advantage_std_floor','learning_rate')})
    opts=actor_contract(cfg)['dual_objective'];options_check(opts)
    assert opts['sampling_seed']==811
    original=editable_objective(cfg);cfg['objective_plugin']['path']='/recovery/plugin'
    assert editable_objective(cfg)==original
    cfg=copy.deepcopy(cfg);cfg['objective_plugin']['numerics']['advantage_std_floor']=.25
    with pytest.raises(ValueError):editable_objective(cfg)
