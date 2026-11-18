"""Real processor/prefetch/session/native transport; model engine is a double."""
from pathlib import Path
import copy

import pytest
import torch

from tests.test_context_attribute_input_candidate import real_inputs, Engine
from tools.reports.context_attribute_actor_candidate import load, REFERENCE_ID
from tools.reports.context_attribute_schedule_candidate import AttributeScheduledBatches, AttributeRolloutSession
from context_reference_cache import ReferenceCache
from context_online_batches import auxiliary_coefficient
from tools.reports.verify_context_attribute_population_v2 import frame_geometry


class SwitchingEngine(Engine):
    def __init__(self):
        super().__init__(REFERENCE_ID)
        self.bundle['contract']={'base_id':'b'*64}

    def switch(self,**spec):
        self.bundle['policy_id']=spec['policy_id']
        return copy.deepcopy(spec)


@pytest.mark.parametrize('arm',['direct','q_input','q_credit'])
def test_real_audio_scheduled_session_native_batch(real_inputs,tmp_path,arm):
    menu,rows,adapters,_,_,_=real_inputs
    adapter=adapters[arm];engine=SwitchingEngine()
    assets=[r[0] for r in rows];targets=[r[1] for r in rows]
    slots=[dict(id=assets[i%3]['id'],padding=i==127,weight=0. if i==127 else 1.) for i in range(128)]
    batches=[dict(epoch=0,index=i,slots=copy.deepcopy(slots)) for i in range(2)]
    cache=ReferenceCache(tmp_path/'reference',policy_id=REFERENCE_ID,base_id='b'*64,contract=engine.bundle['contract'])
    def spec(pid):return dict(policy_id=pid,snapshot='/policy/'+pid,adapter='/adapter/'+pid)
    session=AttributeRolloutSession(engine,cache,input_adapter=adapter,source_root=Path.cwd(),
        reference=spec(REFERENCE_ID),initial_policy=spec(REFERENCE_ID),audio_token_id=151676)
    actor,contract=load(menu,arm,source_root=Path.cwd())
    with AttributeScheduledBatches(batches,assets,targets,adapter,source_root=Path.cwd(),start_update=0,stop_update=2) as stream:
        for step,pid in enumerate((REFERENCE_ID,'updated-policy')):
            items,metadata=stream.consume(step,pid)
            prepared,options,receipt=session.prepare(items,update=step)
            batch=actor.make_batch(prepared,**options)
            payload=batch.non_tensor_batch['multi_modal_inputs']
            assert len(batch)==128 and sum(x['real_request'] for x in payload)==127
            assert receipt['reference_misses']==(3 if step==0 else 0)
            assert all(x['attribute_contract_id']==contract['contract_id'] for x in payload)
            assert all(x['inputs']['input_features'].dtype==torch.bfloat16 for x in payload)
            assert payload[0]['actions'].shape==(1,16,60 if arm=='direct' else 124)
            assert all(x['policy_id']==pid for x in payload)
            assert options['ranking_coefficient']==auxiliary_coefficient(step,634)
            if step==0:session.accept_actor_update(update=step,next_policy=spec('updated-policy'))


def test_actual_fixed_processor_partial_hop_rounding(real_inputs):
    import numpy as np
    import inspect
    from tools.reports.build_context_attribute_candidate import request
    from tools.reports.context_attribute_compact_candidate import encode
    _,_,adapters,processor,_,_=real_inputs
    feature_source=inspect.getsource(type(processor.feature_extractor))
    assert 'rescaled_attention_mask = rescaled_attention_mask[:, :-1]' in feature_source
    for samples in [7999,16000,16001,27648,114672]:
        audio=np.zeros(samples,dtype=np.float32)
        req=request('diagnostic.wav',adapters['direct'].menu,arm='direct')
        inputs,_=encode(processor,req,audio,decision_token_id=151706,audio_token_id=151676,max_tokens=4096)
        frames,width,tokens=frame_geometry(samples)
        assert torch.equal(inputs['input_features_mask'],(torch.arange(width)<frames).int()[None])
        assert inputs['input_features'].shape==(1,128,width)
        assert int((inputs['input_ids']==151676).sum())==tokens
