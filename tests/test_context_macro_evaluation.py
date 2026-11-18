import copy
import json

import pytest
import torch

from test_context_macro_rollout import Client, pool, rows
from multimodal_jev import local_checkpoint
from multimodal_jev.span_request_identity import tensor_identity
from tools.reports import context_macro_evaluation as evaluation


def setup(tmp_path,monkeypatch,n=7):
    monkeypatch.setattr(evaluation,'check_paths',lambda paths:None)
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda paths:None)
    clients=[Client(),Client()]
    engine=pool(tmp_path/'journal',monkeypatch,clients)
    population=[dict(id=str(i),audio_sha256='audio'+str(i),request_sha256='request'+str(i)) for i in range(n)]
    tensors=rows(n)
    def encode(asset):
        row=tensors[int(asset['id'])]
        receipt=dict(asset,candidate_ids=['c'],positions=row['positions'][0].tolist(),
            inputs={k:tensor_identity(v) for k,v in sorted(row['inputs'].items())})
        return row,receipt
    args=dict(candidate_ids=['c'],policy_id='p0',runtime_identity={'profile':'fixture'},
              audio_token_id=9,macro_queries=4,output=tmp_path/'evaluation')
    return engine,population,encode,args,clients


def test_real_pool_full_population_and_short_tail_without_padding(tmp_path,monkeypatch):
    engine,population,encode,args,clients=setup(tmp_path,monkeypatch)
    result=evaluation.evaluate(engine,population,encode,**args)
    assert result['recordings']==7
    assert [b['queries'] for b in result['batches']]==[4,3]
    records=json.loads((args['output']/'predictions.json').read_text())['records']
    assert [r['id'] for r in records]==[r['id'] for r in population]
    assert sorted(i for c in clients for call in c.calls for i in call)==list(range(7))
    assert json.loads((args['output']/'terminal.json').read_text())=={'exit_code':0}
    with pytest.raises(FileExistsError):evaluation.evaluate(engine,population,encode,**args)


@pytest.mark.parametrize('count',[14,31,60])
def test_complete_minds_fsc_slurp_candidate_widths(tmp_path,monkeypatch,count):
    engine,population,encode,args,clients=setup(tmp_path,monkeypatch,n=5)
    candidates=['c'+str(i) for i in range(count)];args['candidate_ids']=candidates
    def expanded(asset):
        row,receipt=encode(asset)
        row['positions']=torch.arange(count).reshape(1,count)
        receipt.update(candidate_ids=candidates,positions=list(range(count)))
        return row,receipt
    for client in clients:
        original=client.score_many
        def score(batch,_original=original,**kw):
            result=_original(batch,**kw)
            for record in result:record.update(prob=[.5]*count,logits=[0.]*count)
            return result
        client.score_many=score
    result=evaluation.evaluate(engine,population,expanded,**args)
    assert result['decisions']==count and result['recordings']==5
    assert [b['queries'] for b in result['batches']]==[4,1]


@pytest.mark.parametrize('fault',['gold','duplicate','stale','runtime'])
def test_preflight_refuses_bad_population_and_policy_without_engine_calls(tmp_path,monkeypatch,fault):
    engine,population,encode,args,clients=setup(tmp_path,monkeypatch)
    if fault=='gold':population[0]['gold']='c'
    if fault=='duplicate':population[1]=copy.deepcopy(population[0])
    if fault=='stale':args['policy_id']='other'
    if fault=='runtime':args['runtime_identity']={'profile':'other'}
    with pytest.raises(ValueError):evaluation.evaluate(engine,population,encode,**args)
    assert not any(c.calls for c in clients)


@pytest.mark.parametrize('fault',['candidate','audio','request','tensor','positions'])
def test_changed_input_receipt_refused_before_dispatch(tmp_path,monkeypatch,fault):
    engine,population,encode,args,clients=setup(tmp_path,monkeypatch)
    def bad(asset):
        row,receipt=encode(asset)
        if fault=='candidate':receipt['candidate_ids']=['other']
        if fault=='audio':receipt['audio_sha256']='other'
        if fault=='request':receipt['request_sha256']='other'
        if fault=='tensor':receipt['inputs']={}
        if fault=='positions':receipt['positions']=[0]
        return row,receipt
    with pytest.raises(ValueError):evaluation.evaluate(engine,population,bad,**args)
    assert not any(c.calls for c in clients)
    assert not (args['output']/'predictions.json').exists()


def test_tail_failure_keeps_first_batch_but_never_publishes_full_score(tmp_path,monkeypatch):
    engine,population,encode,args,clients=setup(tmp_path,monkeypatch)
    def bad(asset):
        if asset['id']=='4':clients[1].bad='timeout'
        return encode(asset)
    with pytest.raises(TimeoutError):evaluation.evaluate(engine,population,bad,**args)
    assert (args['output']/'batch-00000000.json').exists()
    assert not (args['output']/'batch-00000001.json').exists()
    assert not (args['output']/'predictions.json').exists()
    assert not (args['output']/'result.json').exists()
    assert json.loads((args['output']/'terminal.json').read_text())['exit_code']==1


@pytest.mark.parametrize('fault',['digest','coverage','actor','changed_policy'])
def test_completed_receipt_corruption_never_commits_batch(tmp_path,monkeypatch,fault):
    engine,population,encode,args,_=setup(tmp_path,monkeypatch)
    original=engine.score_macro
    def bad(*a,**kw):
        result=original(*a,**kw)
        if fault=='digest':engine.last_receipt['result_sha256']='other'
        if fault=='coverage':engine.last_receipt['completed_indices'].pop()
        if fault=='actor':engine.last_receipt['actor_update_performed']=True
        if fault=='changed_policy':engine.bundle['policy_id']='other'
        return result
    engine.score_macro=bad
    with pytest.raises(ValueError):evaluation.evaluate(engine,population,encode,**args)
    assert not (args['output']/'batch-00000000.json').exists()
