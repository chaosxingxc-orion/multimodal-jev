import threading
from types import SimpleNamespace

import pytest
import torch

import multimodal_jev.cached_instruction_batches as module
from multimodal_jev.cached_span_preparation import PreparedSpanInput


def fixture(monkeypatch):
    owner = threading.get_ident()
    calls = []
    def request(audio):
        return SimpleNamespace(audio=audio, model_input=lambda:dict(audio=audio, version='v1'))
    def example(row, *args):
        assert threading.get_ident() == owner, 'targets constructed in worker'
        return request(row['audio']), dict(id=row['id'], targets=[1], supervised=[True], span_weights=[1.]), None
    monkeypatch.setattr(module, 'compile_intent', lambda audio, *args:request(audio))
    monkeypatch.setattr(module, 'example', example)
    def prepare(job):
        req, digest = job
        assert threading.get_ident() != owner
        calls.append(req.audio)
        return PreparedSpanInput(req.model_input(), {'input_ids':torch.tensor([[1,2]])},
                                 torch.tensor([[1]]), [[3,4]], {'audio':req.audio}, True)
    population = {str(i):dict(id=str(i), audio=f'{i}.wav', audio_sha256='a'*64) for i in range(4)}
    def slot(i,padding=False):
        return dict(id=str(i),view='direct',variant='native',weight=0. if padding else 1.,padding=padding)
    batches = [(0,0,[slot(0),slot(1)]), (0,1,[slot(2),slot(3)]),
               (0,2,[slot(3),slot(3,True)])]
    return population, batches, prepare, calls


def test_resume_late_binding_and_tail(monkeypatch):
    population,batches,prepare,calls=fixture(monkeypatch)
    with module.CachedInstructionBatches(batches,population,{},prepare,start_update=1,
                                        stop_update=3,context_limit=8,global_batch=2) as stream:
        with pytest.raises(ValueError,match='next update'):stream.consume(0,'old')
        items,_=stream.consume(1,'policy1')
        assert stream.last_timing['update_index']==1
        assert stream.last_timing['cache_hits']==stream.last_timing['slots']==2
        assert all(stream.last_timing[k]>=0 for k in ('worker_seconds','wait_seconds','bind_seconds'))
        assert [v['policy_id'] for v in items]==['policy1']*2
        with pytest.raises(ValueError,match='next update'):stream.consume(1,'policy1')
        tail,metadata=stream.consume(2,'policy2')
        assert stream.last_timing['update_index']==2
        assert [v['policy_id'] for v in tail]==['policy2']*2
        assert tail[1]['request_weight']==0 and not tail[1]['real_request']
        assert len({v['request_id'] for v in tail})==2
        assert [v['weight'] for v in metadata]==[1.,0.]
    assert calls==['2.wav','3.wav','3.wav','3.wav']


def test_next_batch_runs_while_consumer_is_busy(monkeypatch):
    population,batches,prepare,_=fixture(monkeypatch)
    second_started=threading.Event();release=threading.Event()
    def gated(job):
        if job[0].audio=='2.wav':
            second_started.set()
            if not release.wait(5):raise RuntimeError('consumer did not release')
        return prepare(job)
    with module.CachedInstructionBatches(batches,population,{},gated,start_update=0,
                                        stop_update=2,context_limit=8,global_batch=2) as stream:
        first,_=stream.consume(0,'p0')
        try:
            assert second_started.wait(5)
            assert len(first)==2
        finally:
            release.set()
        second,_=stream.consume(1,'p1')
        assert second[0]['policy_id']=='p1'


def test_worker_failure_is_not_skipped(monkeypatch):
    population,batches,prepare,_=fixture(monkeypatch)
    def broken(job):
        if job[0].audio=='2.wav':raise OSError('bad audio')
        return prepare(job)
    with module.CachedInstructionBatches(batches,population,{},broken,start_update=0,
                                        stop_update=3,context_limit=8,global_batch=2) as stream:
        stream.consume(0,'p0')
        with pytest.raises(OSError,match='bad audio'):stream.consume(1,'p1')
        assert stream.cursor==1 and stream.queue.closed
