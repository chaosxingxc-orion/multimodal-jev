import copy

import pytest

from multimodal_jev.context_replica_schedule import rebatch, rank_indices
from multimodal_jev.context_training_schedule import build_epoch
from test_context_topology_batch_plan import config


def fixture():
    rows=[dict(id=str(i),role='learning',known_pending_source=False,official_split='train',
               intent=str(i%5),source_group=str(i//2)) for i in range(385)]
    lengths={r['id']:(20+int(r['id'])%13,30+int(r['id'])%19) for r in rows}
    parent=[b for e in range(2) for b in build_epoch(rows,lengths,epoch=e)['batches']]
    c=config();c['source_queries']=len(rows)
    return parent,list(lengths),lengths,c


def test_rebatch_uses_all_real_rows_preserves_order_replaces_old_padding():
    parent,population,lengths,c=fixture();new=rebatch(parent,population,c)
    assert len(new)==6 and [b['real_queries'] for b in new]==[192,192,1]*2
    for e in range(2):
        ids=lambda batches:[s['id'] for b in batches if b['epoch']==e for s in b['slots'] if not s['padding']]
        assert ids(parent)==ids(new)
    for b in new:
        ranks=rank_indices(b,[lengths[s['id']] for s in b['slots']],c)
        assert list(map(len,ranks))==[64,64,64]
        assert sorted(i for rank in ranks for i in rank)==list(range(192))


@pytest.mark.parametrize('fault',['missing','duplicate','weight','epoch','order'])
def test_corrupt_source_cannot_be_rebatched(fault):
    parent,population,_,c=fixture();parent=copy.deepcopy(parent)
    if fault=='missing':parent[0]['slots'].pop()
    if fault=='duplicate':parent[0]['slots'][0]=copy.deepcopy(parent[0]['slots'][1])
    if fault=='weight':parent[0]['slots'][0]['weight']=.5
    if fault=='epoch':parent=[b for b in parent if b['epoch']==0]
    if fault=='order':parent[0],parent[1]=parent[1],parent[0]
    with pytest.raises(ValueError):rebatch(parent,population,c)


@pytest.mark.parametrize('fault',[None,'order','tail_weight','count','topology'])
def test_loader_reconstructs_even_if_modified_files_are_rehashed(tmp_path,monkeypatch,fault):
    import json
    from multimodal_jev import context_training_schedule as parent_module
    from multimodal_jev.context_replica_schedule import load_schedule
    from multimodal_jev.context_topology_batch_plan import derive
    from multimodal_jev.baseline import file_hash
    from multimodal_jev.slurp_training import content_hash
    parent,population,_,c=fixture();batches=rebatch(parent,population,c)
    # Parent authentication has its own tests; reconstructing the new schedule
    # is exercised with actual JSON bytes and hashes here.
    monkeypatch.setattr(parent_module,'load_schedule',lambda *a,**kw:(parent,{}))
    if fault=='order':batches[0]['slots'][0],batches[0]['slots'][1]=batches[0]['slots'][1],batches[0]['slots'][0]
    if fault=='tail_weight':batches[-1]['slots'][-1]['weight']=1.
    path=tmp_path/'batches.json';path.write_text(json.dumps(batches))
    m=dict(schema='jev-replica-source-schedule-v1',release_sha256='release',
        parent_manifest_sha256='parent',parent_release_sha256='oldrelease',
        batch_plan=derive(c),real_population_sha256=content_hash(sorted(population)),
        files={'batches.json':file_hash(path)},total_updates=len(batches),total_real_exposures=770)
    if fault=='count':m['total_real_exposures']+=1
    if fault=='topology':m['batch_plan']['global_batch']=128
    manifest=tmp_path/'MANIFEST.json';manifest.write_text(json.dumps(m))
    kwargs=dict(release_sha256='release',population=population,batch_config=c,
                parent_root='parent',parent_manifest_sha256='parent',parent_release_sha256='oldrelease')
    if fault:
        with pytest.raises(ValueError):load_schedule(tmp_path,file_hash(manifest),**kwargs)
    else:assert load_schedule(tmp_path,file_hash(manifest),**kwargs)[0]==batches
