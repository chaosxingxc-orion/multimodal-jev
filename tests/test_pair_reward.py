from dataclasses import replace
import pytest
from multimodal_jev.task_schema import Candidate,DecisionSchema,JevExample
from multimodal_jev.pair_reward import paired_rewards


def group():
    candidates=tuple(Candidate(k,v) for k,v in [('a','activate'),('b','deactivate'),('c','change')])
    schema=DecisionSchema('choice','choice','Which?',candidates)
    meta={'source_group':'g','audio_sha256':'hash'}
    views=[('native',schema,'a'),('permutation',replace(schema,candidates=candidates[::-1]),'a'),
           ('rebind',replace(schema,candidates=(Candidate('b','activate'),Candidate('a','deactivate'),candidates[2])),'b'),
           ('missing_candidate',replace(schema,candidates=candidates[1:]),None),
           ('noul.True',DecisionSchema('yes','noul','Activate?'),True),
           ('noul.False',DecisionSchema('no','noul','Deactivate?'),False)]
    return [JevExample(v,'train','fixture.wav',s,t,{**meta,'variant':v}) for v,s,t in views]


def test_oracle_and_constant_degenerate_policies():
    rows=group()
    rewards,stats=paired_rewards(rows,[x.target for x in rows])
    assert all(x['pair_bonus']==1 for x in rewards)
    assert all(v['both_correct']==1 for v in stats.values())
    for predictions in [[None]*6,['a']*6,[True]*6]:
        wrong,_=paired_rewards(rows,predictions)
        assert sum(x['total'] for x in wrong)<sum(x['total'] for x in rewards)
    # Same wrong semantic meaning across native/permuted is not rewarded.
    wrong,_=paired_rewards(rows,['c','c','c',None,False,True])
    assert all(x['pair_bonus']==0 for x in wrong)


def test_incomplete_duplicate_and_mispaired_groups_fail():
    rows=group()
    with pytest.raises(ValueError,match='incomplete'):paired_rewards(rows[:-1],[x.target for x in rows[:-1]])
    with pytest.raises(ValueError,match='duplicate'):paired_rewards(rows+rows,[x.target for x in rows]*2)
    rows[1]=replace(rows[1],audio='other.wav')
    with pytest.raises(ValueError,match='audio'):paired_rewards(rows,[x.target for x in rows])


def test_rebind_cannot_silently_replace_distractor_definition():
    rows=group()
    original=rows[2]
    changed=list(original.schema.candidates);changed[-1]=Candidate('c','unrelated new meaning')
    rows[2]=replace(original,schema=replace(original.schema,candidates=tuple(changed)))
    with pytest.raises(ValueError,match='definition set'):
        paired_rewards(rows,[x.target for x in rows])


def test_rollout_pair_bonus_preserves_bad_json_and_aligns_replicates():
    import json
    from types import SimpleNamespace as NS
    from multimodal_jev.paired_rollout import compose_rewards
    examples=[];records=[]
    for replicate in (0,1):
        for ex in group():
            text=json.dumps({'value':ex.target})
            if replicate==1 and ex.metadata['variant']=='missing_candidate':text='not JSON'
            examples.append(ex);records.append(NS(example_id=ex.example_id,replicate=replicate,text=text))
    scores,details=compose_rewards(examples,records,'pair-grpo')
    assert scores[3]>1 and scores[9]==-1
    assert details['pair_bonuses'][9]==0
    order=list(range(len(records)))[::-1]
    reordered,_=compose_rewards([examples[i] for i in order],[records[i] for i in order],'pair-grpo')
    assert reordered==scores[::-1]
    with pytest.raises(ValueError,match='incomplete'):
        compose_rewards(examples[:-1],records[:-1],'pair-grpo')
