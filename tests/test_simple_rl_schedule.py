import json
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.slurp_training import content_hash
from multimodal_jev.simple_rl_schedule import SimpleRLSchedule


def fixture(tmp_path):
    declared=[]
    for epoch in range(2):
        ids=list(range(40513))
        if epoch:ids.reverse()
        slots=[dict(id=str(i),source_group=str(i//3),view='direct',variant='native',weight=1.,padding=False) for i in ids]
        slots += [dict(slots[0],weight=0.,padding=True) for _ in range(63)]
        batches=[slots[i:i+128] for i in range(0,len(slots),128)]
        report=dict(epoch=epoch,updates=317,recordings=40513,padding_slots=63,global_batch=128,world_size=4,
            seed=317,task_mode='direct_final',class_weighting='sqrt_cap4',batches_sha256=content_hash(batches))
        p=tmp_path/f'epoch-{epoch}.json';p.write_text(json.dumps(dict(report=report,batches=batches)))
        declared.append(dict(report,file_sha256=file_hash(p)))
    m=dict(global_batch=128,world_size=4,task_mode='direct_final',class_weighting='sqrt_cap4',
        loss_reduction='real_request_mean',unique_recordings=40513,total_nonpadding_recording_uses=81026,
        manifest_sha256='release',epochs=declared)
    p=tmp_path/'MANIFEST.json';p.write_text(json.dumps(m));return file_hash(p)


def test_full_epochs_boundaries_decay_and_recovery_cursor(tmp_path):
    h=fixture(tmp_path);s=SimpleRLSchedule(tmp_path,manifest_sha256=h,release_sha256='release')
    seen=[]
    for u in range(634):
        slots,meta=s.at(u);seen.extend(v['id'] for v in slots if not v['padding'])
        assert meta['epoch']==u//317 and meta['batch_index']==u%317
    assert len(seen)==81026 and len(set(seen))==40513
    assert s.at(316)[1]['real_requests']==s.at(633)[1]['real_requests']==65
    assert s.at(317)[1]['epoch']==1 and s.at(0)[1]['coefficient']==1.
    assert s.at(633)[1]['coefficient']==.1
    slots,_=s.at(5);slots[0]['weight']=900;assert s.at(5)[0][0]['weight']==1.
    for cursor in (-1,634,True):
        with pytest.raises(ValueError):s.at(cursor)


def test_independent_population_weights_and_learning_membership():
    import math
    from collections import Counter
    rows=[dict(id=str(i),source_group=str(i//3),intent=str(i%60),role='learning',known_pending_source=False) for i in range(40513)]
    counts=Counter(r['intent'] for r in rows)
    raw={c:min(4.,math.sqrt(40513/(60*n))) for c,n in counts.items()}
    norm=sum(raw[c]*n for c,n in counts.items())/40513
    s=SimpleRLSchedule.__new__(SimpleRLSchedule);s.total_updates=634;s.sha256='schedule'
    s.population_hashes=[content_hash(sorted(rows,key=lambda r:r['id']))]*2
    s.identities={r['id']:(r['source_group'],raw[r['intent']]/norm) for r in rows}
    assert s.validate_population(rows)['uses']==81026
    s.identities['0']=(rows[0]['source_group'],1.2)
    with pytest.raises(ValueError,match='weight'):s.validate_population(rows)
    rows[0]['role']='selection'
    with pytest.raises(ValueError,match='learning'):s.validate_population(rows)


@pytest.mark.parametrize('fault',['file_hash','duplicate','moving_weight','early_padding','wrong_role','population','foreign_padding'])
def test_reject_changed_full_schedule_even_with_resealed_manifest(tmp_path,fault):
    h=fixture(tmp_path);p=tmp_path/'epoch-1.json';e=json.loads(p.read_text())
    if fault=='duplicate':e['batches'][0][1]=dict(e['batches'][0][0])
    elif fault=='moving_weight':
        e['batches'][0][0]['weight']=.5;e['batches'][0][1]['weight']=1.5
    elif fault=='early_padding':e['batches'][0][0].update(padding=True,weight=0.)
    elif fault=='wrong_role':e['batches'][0][0]['view']='selection'
    elif fault=='population':e['batches'][0][0]['id']='foreign'
    elif fault=='foreign_padding':e['batches'][-1][-1]['id']='foreign'
    else:e['batches'][0][0]['weight']=2.
    e['report']['batches_sha256']=content_hash(e['batches']);p.write_text(json.dumps(e))
    if fault!='file_hash':
        m=tmp_path/'MANIFEST.json';v=json.loads(m.read_text());v['epochs'][1]=dict(e['report'],file_sha256=file_hash(p))
        m.write_text(json.dumps(v));h=file_hash(m)
    with pytest.raises(ValueError):SimpleRLSchedule(tmp_path,manifest_sha256=h,release_sha256='release')
