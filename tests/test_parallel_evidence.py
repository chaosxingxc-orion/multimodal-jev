import os
from pathlib import Path
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.parallel_evidence import inventory
from tools.reports.context_parallel_batch_audit import audit_batches
from multimodal_jev.context_candidate_task import compile_candidates,bind_candidate_targets
import json


def test_threaded_hashing_rereads_same_size_and_timestamp_changes(tmp_path):
    path=tmp_path/'file';path.write_bytes(b'abcd');st=path.stat()
    first=inventory(tmp_path,io_workers=4)
    assert first=={'file':file_hash(path)}
    path.write_bytes(b'efgh');os.utime(path,ns=(st.st_atime_ns,st.st_mtime_ns))
    assert inventory(tmp_path,io_workers=2)!=first
    path.unlink();path.symlink_to('/etc/hosts')
    with pytest.raises(ValueError,match='symlink'):inventory(tmp_path)


def inputs(tmp_path):
    task=dict(schema={'candidates':[{'id':'a','description':'turn on'},{'id':'b','description':'turn off'}]},thinking={'requirements':['listen']})
    population={'x':dict(id='x',audio='sample.wav',intent='a')}
    request=compile_candidates('sample.wav',[{'id':'a','definition':'turn on'},{'id':'b','definition':'turn off'}],['listen'],semantics='exclusive_event')
    target=bind_candidate_targets(request,semantics='exclusive_event',exclusive_event='a')
    items=[]
    for i in range(4):
        planned=dict(epoch=0,index=i,slots=[dict(id='x',weight=1.,padding=False)],real_queries=1)
        batch=dict(policy_id='p:instruction-update'+str(i),rank_indices=[[0]],
            records=[dict(id='x',request_id=f'x::epoch0-batch{i}-slot0',weight=1.,target=target)])
        path=tmp_path/f'{i}.json';path.write_text(json.dumps(batch));items.append((i,str(path),file_hash(path),planned))
    return items,population,task


def test_processes_preserve_order_and_exact_original_checker(tmp_path):
    items,pop,task=inputs(tmp_path)
    one,_=audit_batches(items,pop,task,policy='p',width=1,world=1,cpu_workers=1)
    many,_=audit_batches(items,pop,task,policy='p',width=1,world=1,cpu_workers=2)
    assert one==many and [r['index'] for r in many]==list(range(4))


@pytest.mark.parametrize('fault',['bytes','target','rank','weight','missing'])
def test_workers_reject_invalid_evidence(tmp_path,fault):
    items,pop,task=inputs(tmp_path);path=Path(items[2][1]);v=json.loads(path.read_text())
    if fault=='missing':path.unlink()
    else:
        if fault=='target':v['records'][0]['target']['labels']=[False,True]
        elif fault=='rank':v['rank_indices']=[[1]]
        elif fault=='weight':v['records'][0]['weight']=0.
        else:v['extra']='modified'
        path.write_text(json.dumps(v))
        if fault!='bytes':items[2]=(2,str(path),file_hash(path),items[2][3])
    with pytest.raises((ValueError,FileNotFoundError)):
        audit_batches(items,pop,task,policy='p',width=1,world=1,cpu_workers=2)
