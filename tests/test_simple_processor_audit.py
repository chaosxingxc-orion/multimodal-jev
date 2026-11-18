import json
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.slurp_training import content_hash
from multimodal_jev.simple_processor_audit import verify_records


def fixture(tmp_path,monkeypatch):
    from multimodal_jev import slurp_intent_structure
    class Request:
        def model_input(self):return {'audio':'wave'}
    monkeypatch.setattr(slurp_intent_structure,'compile_intent',lambda *args:Request())
    rows=[dict(id='one',audio='wave',audio_sha256='bytes')]
    batches=[[dict(id='one',padding=False,view='direct',variant='native')]]
    epochs=[]
    for epoch in range(2):
        p=tmp_path/f'epoch-{epoch}.json';p.write_text(json.dumps(dict(batches=batches)))
        epochs.append(dict(epoch=epoch,file_sha256=file_hash(p),batches_sha256=content_hash(batches)))
    p=tmp_path/'MANIFEST.json';p.write_text(json.dumps(dict(task_mode='direct_final',manifest_sha256='release',
        epochs=epochs,total_nonpadding_recording_uses=2)))
    records=[dict(epoch=e,batch=0,slot=0,id='one',view='direct',variant='native',audio_sha256='bytes',
        candidate_counts=[60],input_sha256=content_hash({'audio':'wave'}),input_tokens=3,positions=[[2]],
        input_ids={'shape':[1,3]},samples=16000,**({'reused':True} if e else {})) for e in range(2)]
    path=tmp_path/'records.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    proof=dict(schedule_sha256=file_hash(p),release_sha256='release',records_sha256=file_hash(path),
        passed=True,context_overflow=[],context_limit=4096,requests=2,unique_input_views=1,
        views={'direct':2},maximum_tokens=3,minimum_tokens=3)
    return rows,path,proof,records


def test_complete_population(tmp_path,monkeypatch):
    rows,path,proof,_=fixture(tmp_path,monkeypatch)
    assert verify_records(tmp_path,path,proof,rows,{})['requests']==2


@pytest.mark.parametrize('fault',['missing','extra','coordinate','wave','reused','position','summary'])
def test_reject_incomplete_or_wrong_input_ledger(tmp_path,monkeypatch,fault):
    rows,path,proof,records=fixture(tmp_path,monkeypatch)
    if fault=='missing':records.pop()
    if fault=='extra':records.append(records[0])
    if fault=='coordinate':records[1]['slot']=1
    if fault=='wave':records[1]['audio_sha256']='other'
    if fault=='reused':records[1]['samples']=16001
    if fault=='position':records[1]['positions']=[[3]]
    if fault=='summary':proof['requests']=1
    path.write_text(''.join(json.dumps(r)+'\n' for r in records));proof['records_sha256']=file_hash(path)
    with pytest.raises(ValueError):verify_records(tmp_path,path,proof,rows,{})
