import copy
from pathlib import Path
import pytest
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.intent_resume import authenticate,validate_lineage


def configs(tmp_path):
    root=tmp_path/'training';suite=tmp_path/'suite'
    previous=dict(model_path='/base',pilot='/pilot',manifest_sha256='manifest',arm='P',global_requests=8,
        seed=317,lr=1e-4,lora_rank=4,lora_alpha=8,lora_execution_dtype='bfloat16',runtime_profile='fixed',step=0)
    current=dict(previous,step=1,previous_root=str(root),previous_suite=str(suite),
        initial_adapter=str(root/'run/adapter'),acceptance_root=str(suite),pins={})
    return previous,current


@pytest.mark.parametrize('key,value',[('arm','D'),('step',3),('step',True),('seed',318),('lr',.001),
    ('manifest_sha256','changed'),('global_requests',4),('lora_execution_dtype','float32'),
    ('initial_adapter','/wrong'),('acceptance_root','/stale')])
def test_lineage_rejects_changed_training_or_stale_policy(tmp_path,key,value):
    previous,current=configs(tmp_path);validate_lineage(previous,current)
    current[key]=value
    with pytest.raises(ValueError):validate_lineage(previous,current)


def fixture(tmp_path):
    previous,current=configs(tmp_path);root=Path(current['previous_root']);suite=Path(current['previous_suite'])
    root.mkdir();suite.mkdir();files={}
    for rank in range(4):
        path=root/'run'/f'rank-{rank}'/'optimizer-local.pt';path.parent.mkdir(parents=True)
        path.write_text('exact shard '+str(rank));files[str(path)]=file_hash(path)
    atomic_json(root/'config.json',previous)
    atomic_json(suite/'config.json',dict(training_root=str(root),adapter=current['initial_adapter'],model='/base',runtime_profile='fixed',adapter_id='adapter'))
    atomic_json(suite/'verified.json',dict(passed=True,root=str(root),suite=str(suite),updates=1,arm='P',
        full_gate_requests=178,exact_export_reload=True,adapter_id='adapter',files=files))
    atomic_json(suite/'ACCEPTED.json',dict(passed=True,adapter_id='adapter',verified_sha256=file_hash(suite/'verified.json')))
    atomic_json(suite/'controller-terminal.json',{'exit_code':0})
    current['pins']={**files,**{str(p):file_hash(p) for p in [root/'config.json',*suite.glob('*.json')]}}
    return current


def test_requires_every_optimizer_shard_and_exact_receipt(tmp_path):
    cfg=fixture(tmp_path)
    assert authenticate(cfg)['previous_updates']==1
    path=Path(cfg['previous_root'])/'run/rank-3/optimizer-local.pt'
    changed=copy.deepcopy(cfg);changed['pins'].pop(str(path))
    with pytest.raises(ValueError,match='not pinned'):authenticate(changed)
    path.write_text('corrupted')
    with pytest.raises(ValueError,match='changed'):authenticate(cfg)


def test_rehashed_failed_controller_still_rejected(tmp_path):
    cfg=fixture(tmp_path);path=Path(cfg['previous_suite'])/'controller-terminal.json'
    atomic_json(path,{'exit_code':1});cfg['pins'][str(path)]=file_hash(path)
    with pytest.raises(ValueError,match='acceptance failed'):authenticate(cfg)
