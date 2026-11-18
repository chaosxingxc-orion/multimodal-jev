import json
import os
from pathlib import Path
import threading
import pytest
from multimodal_jev.audit_singleflight import AuditSingleFlight
from tools.reports import context_shared_source_audit as m
from test_parallel_evidence import inputs


def test_shared_replay_rehashes_bytes_and_keys_task_and_source(tmp_path,monkeypatch):
    items,pop,task=inputs(tmp_path)
    package=tmp_path/'package';collection=tmp_path/'collection';run=collection/'run';run.mkdir(parents=True)
    (package/'source/schedule').mkdir(parents=True)
    (package/'config.json').write_text(json.dumps(dict(experiment_kind='formal_infonce_source',start_update=0,stop_update=4,source_policy_id='p')))
    schedule=[x[3] for x in items]
    (package/'source/schedule/batches.json').write_text(json.dumps(schedule))
    for i,path,_,_ in items:(run/f'batch-{i:04d}.json').write_bytes(Path(path).read_bytes())
    calls=[]
    def parallel(items,pop,task,**kwargs):
        calls.append(1)
        return [dict(index=i,real_queries=1,batch_sha256=h,batch_semantic_sha256=m.identity(json.loads(Path(p).read_text())))
                for i,p,h,s in items],.1
    monkeypatch.setattr(m,'audit_batches',parallel)
    store=AuditSingleFlight();events=[]
    def replay():
        r=m.SharedBatchReplay(package,collection,2,store=store,io_workers=2,events=events,event_lock=threading.Lock())
        for i,p in enumerate(schedule):
            r(json.loads((run/f'batch-{i:04d}.json').read_text()),p,pop,task,policy=f'p:instruction-update{i}',width=1,world=1)
        r.finish()
    replay();replay();assert len(calls)==1 and events[-1]['cache_hit']
    path=run/'batch-0000.json';st=path.stat();raw=path.read_text();path.write_text(raw.replace('1.0','1.1'))
    os.utime(path,ns=(st.st_atime_ns,st.st_mtime_ns));replay()
    assert len(calls)==2 and not events[-1]['cache_hit']
    task['thinking']['requirements'].append('new rule');replay()
    assert len(calls)==3
