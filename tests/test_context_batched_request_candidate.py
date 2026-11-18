import copy
from types import SimpleNamespace as NS

import pytest
import torch

from multimodal_jev.context_resident_rollout import FixedContextRollout
from tools.reports.context_batched_request_candidate import encode_many
from tools.reports.context_batched_owner_candidate import transformed_source
from test_context_rollout_transport import fixture


def case():
    owner, inputs, pos, _ = fixture()
    owner.max_batch_queries = 8; owner.max_batch_tokens = 8192
    second = {k: v.clone() for k, v in inputs.items()}
    second['input_ids'][0, 0] = 7
    second['input_features'] += 3
    rows = [dict(inputs=inputs, positions=pos), dict(inputs=second, positions=pos.clone())]
    calls = []
    def encode(prompts, **kwargs):
        calls.append((prompts, kwargs))
        return [NS(prompt_token_ids=p.extra_kwargs['expected_input_ids'],
                   outputs=NS(data=torch.tensor([[.2], [.8]]))) for p in kwargs['pooling_params']]
    owner.engine = NS(encode=encode)
    return owner, rows, calls


def test_one_encode_and_exact_frozen_wire_identity():
    owner, rows, calls = case()
    results = encode_many(owner, rows, audio_token_id=98, params_class=NS)
    assert len(calls) == 1 and len(calls[0][0]) == 2
    prompts, options = calls[0]
    assert options['lora_request'] is owner.lora
    assert len(options['pooling_params']) == 2
    for row, result, prompt, param in zip(rows, results, prompts, options['pooling_params']):
        single = object.__new__(FixedContextRollout)
        single.bundle=owner.bundle;single.max_model_len=owner.max_model_len
        single.max_decisions=owner.max_decisions;single.lora=owner.lora
        captured=[]
        def engine_encode(ps, **kwargs):
            captured.append((ps, kwargs))
            return [NS(prompt_token_ids=row['inputs']['input_ids'][0].tolist(),
                       outputs=NS(data=torch.tensor([[.2],[.8]])))]
        single.engine=NS(encode=engine_encode)
        reference=single._score(row['inputs'],row['positions'],98,NS)
        assert reference == result
        assert prompt['multi_modal_uuids']['audio'] == [result['request_id']]
        assert param.extra_kwargs == captured[0][1]['pooling_params'].extra_kwargs
        assert prompt['prompt_token_ids'] == captured[0][0][0]['prompt_token_ids']
        for key, value in prompt['multi_modal_data']['audio'].items():
            assert torch.equal(value, captured[0][0][0]['multi_modal_data']['audio'][key])


@pytest.mark.parametrize('fault', ['empty', 'query_budget', 'token_budget', 'duplicate', 'label',
    'second_marker', 'second_nan', 'second_frame_mask', 'audio_marker', 'positions_type', 'negative_id'])
def test_full_batch_validated_before_any_model_call(fault):
    owner, rows, calls = case()
    audio=98
    if fault=='empty': rows=[]
    if fault=='query_budget': owner.max_batch_queries=1
    if fault=='token_budget': owner.max_batch_tokens=13
    if fault=='duplicate': rows[1]=copy.deepcopy(rows[0])
    if fault=='label': rows[1]['inputs']['labels']=torch.tensor([1])
    if fault=='second_marker': rows[1]['positions'][0,1]=5
    if fault=='second_nan': rows[1]['inputs']['input_features'][0,0,0]=float('nan')
    if fault=='second_frame_mask': rows[1]['inputs']['input_features_mask'][0,1]=0
    if fault=='audio_marker': audio=99
    if fault=='positions_type': rows[1]['positions']=rows[1]['positions'].float()
    if fault=='negative_id': rows[1]['inputs']['input_ids'][0,0]=-1
    with pytest.raises(ValueError): encode_many(owner, rows, audio_token_id=audio, params_class=NS)
    assert not calls


@pytest.mark.parametrize('fault', ['missing', 'reorder', 'shape', 'nan', 'range', 'policy'])
def test_result_failures_reject_whole_batch(fault):
    owner, rows, calls=case(); original=owner.engine.encode
    def encode(*args,**kwargs):
        result=original(*args,**kwargs)
        if fault=='missing':result.pop()
        if fault=='reorder':result.reverse()
        if fault=='shape':result[1].outputs.data=torch.ones(1)
        if fault=='nan':result[1].outputs.data[0,0]=float('nan')
        if fault=='range':result[1].outputs.data[0,0]=2
        if fault=='policy':owner.bundle=dict(owner.bundle,policy_id='b'*64)
        return result
    owner.engine.encode=encode
    with pytest.raises(ValueError):encode_many(owner,rows,audio_token_id=98,params_class=NS)
    assert len(calls)==1


def test_source_derivation_preserves_switch_lifecycle_and_rejects_drift(tmp_path):
    from pathlib import Path
    source=transformed_source(Path(__file__).resolve().parents[1])
    compile(source,'<candidate>','exec')
    original=Path('tools/jev/context_switch_rollout.py').read_text()
    assert source[source.index('    def switch('):] == original[original.index('    def switch('):]
    assert "max_num_seqs=config['max_batch_queries']" in source
    assert 'BatchedContextWorkerCandidate' in source
    p=tmp_path/'tools/jev/context_switch_rollout.py';p.parent.mkdir(parents=True);p.write_text(original+'\n')
    with pytest.raises(ValueError,match='source changed'):transformed_source(tmp_path)
