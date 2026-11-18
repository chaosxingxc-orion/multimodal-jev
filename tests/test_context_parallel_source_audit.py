import copy
import json
from pathlib import Path
import pytest
from tools.reports import context_parallel_source_audit as m
from test_parallel_evidence import inputs


@pytest.mark.parametrize('fault',[None,'target','schedule','policy','population','incomplete','late_mutation'])
def test_parallel_proof_consumption_rejects_mismatch(tmp_path,monkeypatch,fault):
    items,pop,task=inputs(tmp_path)
    package=tmp_path/'package';collection=tmp_path/'collection';run=collection/'run';run.mkdir(parents=True)
    (package/'source/schedule').mkdir(parents=True)
    (package/'config.json').write_text(json.dumps(dict(experiment_kind='formal_infonce_source',start_update=0,stop_update=4,source_policy_id='p')))
    planned=[item[3] for item in items]
    (package/'source/schedule/batches.json').write_text(json.dumps(planned))
    for i,path,_,_ in items:(run/f'batch-{i:04d}.json').write_bytes(Path(path).read_bytes())
    # Use deterministic results for consumption faults; actual worker execution
    # and rejection are covered by test_parallel_evidence.
    def parallel(items,*args,**kwargs):
        return [dict(index=i,real_queries=1,batch_sha256=digest,batch_semantic_sha256=m.identity(json.loads(Path(path).read_text())))
                for i,path,digest,p in items],.1
    monkeypatch.setattr(m,'audit_batches',parallel)
    replay=m.BatchReplay(package,collection,2)
    def execute():
        for i in range(4):
            batch=json.loads((run/f'batch-{i:04d}.json').read_text());p=copy.deepcopy(planned[i]);policy=f'p:instruction-update{i}'
            pp=pop
            if i==1:
                if fault=='target':batch['records'][0]['target']['labels']=[False,True]
                if fault=='schedule':p['real_queries']=0
                if fault=='policy':policy='wrong'
                if fault=='population':pp=copy.deepcopy(pop)
                if fault=='incomplete':break
            replay(batch,p,pp,task,policy=policy,width=1,world=1)
        if fault=='late_mutation':pop['x']['intent']='b'
        return replay.finish()
    if fault:
        with pytest.raises(ValueError):execute()
    else:assert execute()['batches']==4
