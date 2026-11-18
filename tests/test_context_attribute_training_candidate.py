import ast
import copy
import json
from pathlib import Path
import types

import pytest

from tools.reports.context_attribute_training_candidate import (
    derive,load_dataset,validate_targets,AttributeTrainingRuntime)

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'.local/decision-objective-research'
DATA=BASE/'slurp-native-attribute-prob-candidate-v1'


def test_exact_trainer_reuse_and_bounded_lifecycle():
    source,receipt=derive(ROOT)
    module=types.ModuleType('test_attribute_trainer')
    exec(compile(source,'<derived>','exec'),module.__dict__)
    assert callable(module.run) and receipt['substitutions']==7
    assert receipt['training_dispatch_allowed'] is False
    # Unchanged training, export and restore blocks must remain byte-identical.
    original=(ROOT/'tools/jev/train_context_online_candidate.py').read_text()
    for begin,end in [
        ('        if start_update:', '        # Bind resumed/live actor'),
        ('                actor=build_actor', '                record=dict(update='),
        ('        if c[\'engineering_prefix\'] is not True', '        contract=')]:
        segment=original[original.index(begin):original.index(end,original.index(begin))]
        assert segment in source
    # No changes to original source/module globals, no device access on import.
    assert original==(ROOT/'tools/jev/train_context_online_candidate.py').read_text()
    assert not hasattr(module,'ATTRIBUTE_RUNTIME')
    ast.parse(source)


def test_changed_upstream_trainer_rejected(tmp_path):
    p=tmp_path/'tools/jev/train_context_online_candidate.py';p.parent.mkdir(parents=True)
    p.write_text((ROOT/'tools/jev/train_context_online_candidate.py').read_text()+'\n')
    with pytest.raises(ValueError):derive(tmp_path)


def test_full_real_train_target_and_audio_binding():
    menu,inputs,targets,manifest=load_dataset(DATA)
    source=Path(manifest['release'])/'learning.jsonl'
    rows=[json.loads(s) for s in source.read_text().splitlines()]
    population={r['id']:r for r in rows}
    validate_targets(menu,inputs,targets,population)
    assert len(population)==40513
    # Only copied fixtures are changed; sealed dataset stays untouched.
    small_inputs=inputs[:1];small_targets=copy.deepcopy(targets[:1])
    small_population={inputs[0]['id']:population[inputs[0]['id']]}
    small_targets[0]['labels'][0]=not small_targets[0]['labels'][0]
    with pytest.raises(ValueError):validate_targets(menu,small_inputs,small_targets,small_population)
    small_targets=copy.deepcopy(targets[:1]);small_targets[0]['supervised'][0]=False
    with pytest.raises(ValueError):validate_targets(menu,small_inputs,small_targets,small_population)
    bad=copy.deepcopy(small_population);bad[inputs[0]['id']]['audio_sha256']='wrong'
    with pytest.raises(ValueError):validate_targets(menu,small_inputs,targets[:1],bad)
    with pytest.raises(ValueError):validate_targets(menu,small_inputs,targets[:1]+targets[:1],small_population)


def test_runtime_binds_full_dataset_and_preserves_bounded_contract(monkeypatch):
    # Full real release/targets, synthetic grouping metadata: publication still
    # requires the real processor audit and is tested separately.
    import tools.reports.context_attribute_training_candidate as candidate
    from tests.test_context_attribute_training_contract import manifest
    menu,_,_,_=load_dataset(DATA)
    c=json.loads((BASE/'context-rl-grpo-arithmetic-v2-package/config.json').read_text())
    c.update(attribute_source_root=str(ROOT),attribute_dataset=str(DATA),attribute_arm='q_credit',
        attribute_schedule='/unpublished-candidate',attribute_schedule_sha256='a'*64,
        engineering_prefix=True,training_dispatch_allowed=False,start_update=0,stop_update=2,
        checkpoint_interval=1,pins={})
    batches=[]
    for epoch in range(2):
        batches+=json.loads((BASE/f'context-rl-grpo-arithmetic-v2-package/source/schedule/epoch-{epoch}.json').read_text())['batches']
    schedule=manifest(menu,c,'q_credit')
    schedule.update(candidate_manifest_sha256=candidate.DATA_SHA,source_schedule_sha256=c['schedule_sha256'])
    monkeypatch.setattr(candidate,'load_attribute_schedule',lambda *a,**kw:(copy.deepcopy(batches),copy.deepcopy(schedule)))
    runtime=AttributeTrainingRuntime(c)
    assert len(runtime.population)==40513 and len(runtime.targets)==40513
    assert runtime.contract['reward']=='Final_set_F1+0.25*((scenario_set_F1+action_set_F1)/2)'
    assert runtime.contract['dataset_manifest_sha256']==candidate.DATA_SHA
    assert runtime.contract['attribute']['arithmetic']==c['binary_policy_arithmetic']
    loaded,_=runtime.full_schedule(runtime.population)
    assert len(loaded)==634
    loaded[0]['slots'][0]['weight']=123
    assert runtime.batches[0]['slots'][0]['weight']==1.
    with pytest.raises(ValueError):runtime.full_schedule({})


@pytest.mark.parametrize('fault',['formal','budget','bounds','arithmetic','devices'])
def test_unsafe_runtime_rejected_before_inputs_or_device(fault):
    c=json.loads((BASE/'context-rl-grpo-arithmetic-v2-package/config.json').read_text())
    c.update(attribute_source_root=str(ROOT),engineering_prefix=True,training_dispatch_allowed=False,
        start_update=0,stop_update=2,checkpoint_interval=1)
    if fault=='formal':c['training_dispatch_allowed']=True
    if fault=='budget':c['stop_update']=634
    if fault=='bounds':c['behavior_bounds']['probability_max_abs']=.1
    if fault=='arithmetic':c['binary_policy_arithmetic']='old'
    if fault=='devices':c['physical_devices']=[0,1,2,3]
    with pytest.raises(ValueError):AttributeTrainingRuntime(c)
