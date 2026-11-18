import json
from pathlib import Path
import pytest
from multimodal_jev.context_training_schedule import load_schedule
from multimodal_jev.slurp_training import content_hash
from multimodal_jev.baseline import file_hash


def build(tmp_path,fault=None):
    population={str(i):{} for i in range(65)};reports=[]
    for epoch in range(2):
        slots=[dict(id=str(i),weight=1.,padding=False) for i in range(65)]
        slots += [dict(id=str(i),weight=0.,padding=True) for i in range(63)]
        ranks=[list(range(r*32,(r+1)*32)) for r in range(4)]
        if epoch==1:
            if fault=='duplicate':slots[1]=dict(slots[0])
            if fault=='weight':slots[0]['weight']=2.
            if fault=='rank':ranks[0][0]=1
        batches=[dict(epoch=epoch,index=0,slots=slots,rank_indices=ranks)]
        path=tmp_path/f'epoch-{epoch}.json';path.write_text(json.dumps(dict(batches=batches)))
        reports.append(dict(updates=1,batches_sha256=content_hash(batches),file_sha256=file_hash(path)))
    m=dict(schema='context-full-learning-schedule-v1',release_sha256='a'*64,global_batch=128,
        world_size=4,microbatch=8,accumulation=4,epochs=reports,total_updates_per_stage=2)
    p=tmp_path/'MANIFEST.json';p.write_text(json.dumps(m))
    return population,file_hash(p)


def test_tail_rank_population_verified(tmp_path):
    population,h=build(tmp_path)
    batches,_=load_schedule(tmp_path,h,release_sha256='a'*64,population=population)
    assert len(batches)==2


@pytest.mark.parametrize('fault',['duplicate','weight','rank'])
def test_self_consistent_hashes_cannot_hide_bad_schedule(tmp_path,fault):
    population,h=build(tmp_path,fault)
    with pytest.raises(ValueError):load_schedule(tmp_path,h,release_sha256='a'*64,population=population)
