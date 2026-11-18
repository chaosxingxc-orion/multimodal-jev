from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_context_macro_formal_candidate import schedule
from tools.reports import context_macro_tail_candidate as m
from tools.reports.context_macro_trainer_candidate import generate as original
from multimodal_jev.cached_context_batches import CachedContextBatches
from multimodal_jev.encoded_audio_cache import canonical


def config():
    return dict(experiment_kind='tail_admission',start_update=0,stop_update=2,microbatch=8,
        global_batch=128,engineering_prefix=True,training_dispatch_allowed=False,
        checkpoint_interval=1,stage='task')


def test_generated_entry_keeps_native_update_and_full_schedule_authentication():
    before=original(Path.cwd());c=m.contract(config());source=m.generate(Path.cwd())
    assert c['real_queries']==[65,128] and c['formal_history'] is False
    assert 'actor.update_policy(batch)' in source and 'session.prepare(items,update=index)' in source
    assert 'batches=select_tail(batches,population.keys())' in source
    assert "contract['tail_admission']=tail" in source and 'tail_admission=tail' in source
    assert original(Path.cwd())==before


@pytest.mark.parametrize('change',[{'experiment_kind':'efficiency'},{'start_update':316},
    {'stop_update':100},{'microbatch':16},{'global_batch':192},{'resume_root':'/state'},
    {'training_dispatch_allowed':True},{'stage':'instruction'}])
def test_tail_diagnostic_cannot_be_mistaken_for_formal_training(change):
    with pytest.raises(ValueError):m.contract(dict(config(),**change))


def test_real_loader_preserves_tail_truth_zero_weights_and_original_query_identity():
    batches,ids=schedule();chosen=m.select(batches,ids)
    population={uid:dict(id=uid,audio=uid+'.wav',audio_sha256='a'*64,intent='a') for uid in ids}
    task=dict(schema=dict(candidates=[dict(id='a',description='first'),dict(id='b',description='second')]),
        thinking=dict(requirements=['Listen to audio']))
    def prepare(pair):
        request,_=pair
        return SimpleNamespace(request=canonical(request.model_input()),cache_hit=True,
            inputs=dict(input_ids=torch.tensor([[1,2]])),positions=torch.tensor([[0,1]]),audio_receipt={})
    with CachedContextBatches(chosen,population,task,prepare,start_update=0,stop_update=2) as stream:
        tail,metadata=stream.consume(0,'p0');full,_=stream.consume(1,'p1')
    assert sum(x['real_request'] for x in tail)==65
    assert sum(x['request_weight'] for x in tail)==65
    assert len({x['request_id'] for x in tail})==128
    assert all('epoch0-batch316' in x['request_id'] for x in tail)
    assert all('epoch1-batch0' in x['request_id'] for x in full)
    assert sum(x['real_request'] for x in full)==128
    assert all(x['labels'].tolist()==[[True,False]] for x in tail)
    assert batches[316] is chosen[0] and batches[317] is chosen[1]
