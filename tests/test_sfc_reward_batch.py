from copy import deepcopy
import json
import random
import pytest

from multimodal_jev.sfc_reward_batch import compose,digest
from test_sfc_teacher_feedback import fixture as input_fixture


class Tokenizer:
    def __init__(self):self.values={}
    def encode(self,text):
        key=len(self.values);self.values[key]=text;return [key,999]
    def __len__(self):return 1000
    def decode(self,ids,skip_special_tokens):return self.values[ids[0]] if ids[-1]==999 else 'wrong'


def fixture():
    rows,gold,partition=input_fixture();requests=rows[:2];targets=gold[:2];records=[];evidence={};tok=Tokenizer()
    for r,g in zip(requests,targets):
        for d in r['input']['decisions']:
            key=(r['id'],d['id']);evidence[key]=dict(prompt_sha256=digest(key),input_ids_sha256=digest(['expanded',key]))
            for rep in range(2):
                text=json.dumps(g['answers'][d['id']]);ids=tok.encode(text)
                records.append(dict(example_id=r['id'],decision_id=d['id'],request_sha256=digest(r),replicate=rep,
                    policy_sha256='a'*64,instruction_sha256='b'*64,**evidence[key],raw_text=text,
                    token_ids=ids,log_probs=[-.4,-.2],finish_reason='stop'))
    return dict(all_requests=rows,partition=partition,requests=requests,targets=targets,records=records,
        prompt_evidence=evidence,policy_sha256='a'*64,instruction_sha256='b'*64,group_size=2,tokenizer=tok,
        mode='function-conjunction')


def test_prompt_groups_separate_turns_and_decisions_but_share_replicates():
    args=fixture();result=compose(**args)
    assert len(result['rewards'])==24 and result['prompt_groups']==12 and result['source_groups']==1
    assert all(v==1.2 for v in result['rewards'])
    assert len(set(result['uid']))==12 and all(result['uid'].count(uid)==2 for uid in result['uid'])
    assert result['uid'][0]==result['uid'][1] and result['uid'][0]!=result['uid'][2]
    assert result['uid'][0]!=result['uid'][12]  # Same conversation, different turn.


def test_arrival_order_does_not_change_rewards_or_pairing():
    args=fixture();record=args['records'][3]
    record['raw_text']='true';record['token_ids']=args['tokenizer'].encode('true')
    before=compose(**args)
    expected={(r['example_id'],r['decision_id'],r['replicate']):(v,u) for r,v,u in
              zip(args['records'],before['rewards'],before['uid'])}
    random.Random(17).shuffle(args['records']);after=compose(**args)
    actual={(r['example_id'],r['decision_id'],r['replicate']):(v,u) for r,v,u in
            zip(args['records'],after['rewards'],after['uid'])}
    assert actual==expected
    # Only replicate 1 of the first function loses its conjunction bonus.
    uid=args['requests'][0]['id']
    assert actual[uid,'f00.calls',0][0]==1.2 and actual[uid,'f00.calls',1][0]==1.0


@pytest.mark.parametrize('change',['missing','duplicate','policy','instruction','prompt','expanded','request',
    'replicate','bool_replicate','empty_tokens','bad_token','nan_logp','positive_logp','text','finish','extra_field',
    'missing_turn','selection','partition','evidence','duplicate_target','group_size'])
def test_bad_sampling_or_data_boundaries_rejected(change):
    args=fixture();records=args['records'];r=records[0]
    if change=='missing':records.pop()
    if change=='duplicate':records.append(deepcopy(r))
    for kind,field in [('policy','policy_sha256'),('instruction','instruction_sha256'),('prompt','prompt_sha256'),
                       ('expanded','input_ids_sha256'),('request','request_sha256')]:
        if change==kind:r[field]='c'*64
    if change=='replicate':r['replicate']=2
    if change=='bool_replicate':r['replicate']=True
    if change=='empty_tokens':r['token_ids']=[]
    if change=='bad_token':r['token_ids']=[True,999]
    if change=='nan_logp':r['log_probs'][0]=float('nan')
    if change=='positive_logp':r['log_probs'][0]=.1
    if change=='text':r['raw_text']='different'
    if change=='finish':r['finish_reason']='abort'
    if change=='extra_field':r['target']='leak'
    if change=='missing_turn':args['requests']=args['requests'][:1];args['targets']=args['targets'][:1]
    if change=='selection':args['requests']=args['all_requests'][2:]
    if change=='partition':args['partition']['learning']+=args['partition']['selection']
    if change=='evidence':args['prompt_evidence'].pop(next(iter(args['prompt_evidence'])))
    if change=='duplicate_target':args['targets'].append(args['targets'][0])
    if change=='group_size':args['group_size']=1
    with pytest.raises(ValueError):compose(**args)


def test_malformed_outputs_retained_in_groups_and_rewards():
    args=fixture()
    for r in args['records']:
        r['raw_text']='```json\nfalse\n```';r['token_ids']=args['tokenizer'].encode(r['raw_text']);r['finish_reason']='length'
    result=compose(**args)
    assert len(result['rewards'])==24 and all(v==-1 for v in result['rewards'])
    assert result['prompt_groups']==12 and result['response_tokens']==48
