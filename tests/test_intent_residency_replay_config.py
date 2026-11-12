from pathlib import Path
import copy
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from multimodal_jev.baseline import atomic_json,file_hash
from probe_intent_residency_replay_npu import validate


def fixture(tmp_path):
    reference=tmp_path/'reference';reference.mkdir()
    config=dict(arm='D',step=1,global_requests=8,seed=317,lr=1e-4,lora_execution_dtype='bfloat16',
                initial_adapter='/accepted/adapter',acceptance_root='/accepted/gate',pins={})
    atomic_json(reference/'config.json',config);atomic_json(reference/'launch.json',{'container_id':'reference'})
    paths=[reference/'config.json',reference/'launch.json']
    for rank in range(4):
        path=reference/'run'/f'rank-{rank}'/'optimizer-local.pt';path.parent.mkdir(parents=True);path.write_bytes(b'pinned-placeholder');paths.append(path)
    return dict(config,replay_reference=str(reference),replay_reference_suite='/reference-suite',pins={str(p):file_hash(p) for p in paths})


def test_exact_replay_config_passes_and_all_rank_pins_required(tmp_path):
    c=fixture(tmp_path);validate(c)
    c['pins'].pop(next(p for p in c['pins'] if 'rank-3' in p))
    with pytest.raises(ValueError,match='shards'):validate(c)


@pytest.mark.parametrize('key,value',[('lr',2e-4),('arm','P'),('step',0),('global_requests',32),('initial_adapter','/other'),('acceptance_root','/other'),('adam_betas',[0.,.999])])
def test_replay_cannot_change_learning_factors(tmp_path,key,value):
    c=fixture(tmp_path);c[key]=value
    with pytest.raises(ValueError):validate(c)
