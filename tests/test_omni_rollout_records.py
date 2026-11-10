from types import SimpleNamespace as NS

import pytest

from multimodal_jev.omni_rollout_records import capture, sampling_kwargs
from multimodal_jev.sfc_reward_batch import compose, digest
from test_sfc_reward_batch import fixture


class Ids(list):
    def tolist(self): return list(self)


def setup():
    args = fixture()
    request = args['requests'][0]
    decision = request['input']['decisions'][0]['id']
    records = args['records'][:2]
    prepared = dict(request_sha256=digest(request), input_ids=Ids([12, 13]),
                    input_ids_sha256=digest([12, 13]), prompt_sha256=digest('prompt'))
    response = NS(finished=True, prompt_token_ids=[12, 13], outputs=[
        NS(index=r['replicate'], token_ids=r['token_ids'], text=r['raw_text'],
           finish_reason='stop', logprobs=[{t:NS(logprob=p)} for t,p in zip(r['token_ids'],r['log_probs'])])
        for r in records])
    kwargs = dict(policy_sha256=args['policy_sha256'], instruction_sha256=args['instruction_sha256'], group_size=2)
    return args, request, decision, prepared, response, kwargs


def test_complete_capture_joins_existing_reward_composition():
    args, row, decision, prepared, response, kwargs = setup()
    response.outputs.reverse()
    records = capture(response, row, decision, prepared, args['tokenizer'], **kwargs)
    assert [r['replicate'] for r in records] == [0,1]
    args['records'][:2] = records
    args['prompt_evidence'][row['id'],decision] = {k:prepared[k] for k in ('prompt_sha256','input_ids_sha256')}
    report = compose(**args)
    assert report['prompt_groups'] == 12 and len(report['rewards']) == 24
    assert report['uid'][0] == report['uid'][1]


@pytest.mark.parametrize('change', ['unfinished','prompt','missing','duplicate','empty','text','logprobs',
    'token_missing','nan','positive','logprob_count','prepared','policy','reason','budget','bool_index'])
def test_capture_rejects_mismatched_or_incomplete_engine_output(change):
    args, row, decision, prepared, response, kwargs = setup()
    a=response.outputs[0]
    if change=='unfinished': response.finished=False
    if change=='prompt': response.prompt_token_ids.reverse()
    if change=='missing': response.outputs.pop()
    if change=='duplicate': response.outputs[1].index=0
    if change=='empty': a.token_ids=[]
    if change=='text': a.text='repaired'
    if change=='logprobs': a.logprobs=None
    if change=='token_missing': a.logprobs[0]={}
    if change=='nan': a.logprobs[0][a.token_ids[0]].logprob=float('nan')
    if change=='positive': a.logprobs[0][a.token_ids[0]].logprob=0.1
    if change=='logprob_count': a.logprobs.pop()
    if change=='prepared': prepared['input_ids'][0]=14
    if change=='policy': kwargs['policy_sha256']='unbound'
    if change=='reason': a.finish_reason='abort'
    if change=='budget': kwargs['max_tokens']=1
    if change=='bool_index': a.index=True
    with pytest.raises(ValueError): capture(response,row,decision,prepared,args['tokenizer'],**kwargs)


def test_sampling_policy_matches_unmodified_actor_distribution():
    p=sampling_kwargs(2,17)
    assert p['temperature']==1 and p['top_p']==1 and p['top_k']==0 and p['min_p']==0
    assert p['logprobs']==0 and p['repetition_penalty']==1
    assert p['presence_penalty']==p['frequency_penalty']==0
    assert p['min_tokens']==0 and p['stop'] is None and not p['ignore_eos']
    for args in [(1,17),(True,17),(2,-1),(2,17,0),(2,17,1025)]:
        with pytest.raises(ValueError): sampling_kwargs(*args)
