import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from multimodal_jev.baseline import atomic_json,file_hash
from run_intent_bounded_continuation import validate


def fixture(tmp_path):
    cfg=dict(steps=[],pins={},initial_updates=2,total_updates=4)
    previous=tmp_path/'step2';previous.mkdir();suite=tmp_path/'suite2'
    parent=dict(step=1,model_path='model',pilot='pilot',manifest_sha256='hash',arm='D',global_requests=8,
        seed=317,lr=1e-4,lora_rank=4,lora_alpha=8,lora_execution_dtype='bfloat16',runtime_profile='v2')
    atomic_json(previous/'config-template.json',parent)
    cfg.update(previous_training=str(previous),previous_suite=str(suite))
    for n in (3,4):
        root=tmp_path/f'step{n}';new_suite=tmp_path/f'suite{n}';root.mkdir();new_suite.mkdir()
        current=dict(parent,step=n-1,previous_root=str(previous),previous_suite=str(suite),
            initial_adapter=str(previous/'run/adapter'),acceptance_root=str(suite))
        atomic_json(root/'config-template.json',current)
        atomic_json(new_suite/'config-template.json',dict(training_root=str(root)))
        for base in (root,new_suite):
            atomic_json(base/'prepared.json',{});atomic_json(base/'source-manifest.json',{})
            for name in ('prepared.json','config-template.json','source-manifest.json'):
                cfg['pins'][str(base/name)]=file_hash(base/name)
        cfg['steps'].append(dict(training_root=str(root),suite_root=str(new_suite)))
        previous,suite,parent=root,new_suite,current
    return cfg


def test_accepts_contiguous_fixed_budget(tmp_path):
    validate(fixture(tmp_path))


@pytest.mark.parametrize('change',['skip','budget','cursor','reused','unpinned','reward_arm'])
def test_rejects_silent_budget_or_lineage_changes(tmp_path,change):
    c=fixture(tmp_path)
    if change=='skip':c['steps'].reverse()
    if change=='budget':c['total_updates']=5
    if change=='cursor':c['initial_updates']=1
    if change=='reused':c['steps'][1]=copy.deepcopy(c['steps'][0])
    if change=='unpinned':c['pins'].pop(next(iter(c['pins'])))
    if change=='reward_arm':
        p=Path(c['steps'][1]['training_root'])/'config-template.json'
        import json
        x=json.loads(p.read_text());x['arm']='P';atomic_json(p,x);c['pins'][str(p)]=file_hash(p)
    with pytest.raises(ValueError):validate(c)
