import threading
import pytest
import torch

import multimodal_jev.cached_context_batches as module
from multimodal_jev.cached_context_preparation import PreparedContextInput
from multimodal_jev.encoded_audio_cache import canonical


def fixture(monkeypatch):
    main=threading.get_ident();original=module.bind_candidate_targets
    def bind(*a,**kw):
        assert threading.get_ident()==main
        return original(*a,**kw)
    monkeypatch.setattr(module,'bind_candidate_targets',bind)
    task=dict(schema=dict(candidates=[dict(id='a',description='first event'),dict(id='b',description='second event')]),thinking=dict(requirements=['Listen to audio']))
    population={str(i):dict(id=str(i),audio=f'{i}.wav',audio_sha256='a'*64,intent='a' if i%2==0 else 'b') for i in range(3)}
    def slot(i,pad=False):return dict(id=str(i),weight=0. if pad else 1.,padding=pad)
    batches=[dict(epoch=0,index=0,slots=[slot(0),slot(1)]),dict(epoch=0,index=1,slots=[slot(2),slot(0,True)])]
    def prepare(job):
        assert threading.get_ident()!=main
        request,_=job
        assert 'intent' not in request.model_input()
        return PreparedContextInput(canonical(request.model_input()),dict(input_ids=torch.tensor([[99,99]])),torch.tensor([[0,1]]),{},True)
    return population,task,batches,prepare


def test_resume_padding_policy_and_labels_bind_after_prefetch(monkeypatch):
    population,task,batches,prepare=fixture(monkeypatch)
    with module.CachedContextBatches(batches,population,task,prepare,start_update=1,stop_update=2,global_batch=2) as stream:
        assert all(set(v)=={'audio','audio_sha256'} for v in stream.assets.values())
        with pytest.raises(ValueError):stream.consume(0,'stale')
        items,metadata=stream.consume(1,'current')
        assert len({x['request_id'] for x in items})==2
        assert [x['policy_id'] for x in items]==['current']*2
        assert items[0]['labels'].tolist()==[[True,False]]
        assert not items[1]['real_request'] and items[1]['request_weight']==0
        assert stream.last_timing['cache_hits']==2
        with pytest.raises(ValueError):stream.consume(1,'current')


def test_background_failure_does_not_skip_cursor(monkeypatch):
    population,task,batches,prepare=fixture(monkeypatch)
    def broken(job):
        if job[0].audio=='2.wav':raise OSError('fixture failed')
        return prepare(job)
    with module.CachedContextBatches(batches,population,task,broken,start_update=0,stop_update=2,global_batch=2) as stream:
        stream.consume(0,'p0')
        with pytest.raises(OSError):stream.consume(1,'p1')
        assert stream.cursor==1 and stream.queue.closed
