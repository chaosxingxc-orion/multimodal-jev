from types import SimpleNamespace as NS
import pytest
from multimodal_jev.advantage_summary import summarize_signal, summarize_sequence_coefficients


def fixture():
    examples=[NS(example_id=uid,split='train',metadata={'source_group':group,'variant':view})
              for uid,group,view in [('a','x','native'),('b','x','rebind'),('c','y','native')]]
    rows=[{'example_id':ex.example_id,'replicate':rep,'grpo':-1.,'pair_grpo':-1.}
          for ex in examples for rep in range(2)]
    rows[0]['pair_grpo']=1.;rows[2]['pair_grpo']+=1e-7
    audit={'status':'passed','replicates':2,'responses':6,'per_response':rows,
           'pair_bonuses':[.2,0,.1,0,0,0],
           'center_only_control':{'grpo':[0.]*6,'pair_grpo':[.1,-.1,.05,-.05,0.,0.]}}
    return audit,examples


def test_material_coverage_counts_unique_prompts_recordings_and_response_signs():
    audit,examples=fixture();r=summarize_signal(audit,examples)
    assert r['overall']['responses']==6 and r['overall']['prompts']==3
    assert r['overall']['standard']['material_responses']==1
    assert r['overall']['standard']['material_recordings']==1
    assert r['overall']['standard_sign_change_responses']==1
    assert r['overall']['center_only']['material_responses']==4
    assert r['overall']['center_only']['material_prompts']==2
    assert r['by_recording']['y']['center_only']['material_responses']==0
    assert r['by_variant']['rebind']['standard']['material_responses']==0


@pytest.mark.parametrize('change',['duplicate','missing','nan','split'])
def test_incomplete_or_untrustworthy_audit_rejected(change):
    audit,examples=fixture()
    if change=='duplicate':audit['per_response'][1]=audit['per_response'][0]
    elif change=='missing':audit['pair_bonuses'].pop()
    elif change=='nan':audit['center_only_control']['grpo'][0]=float('nan')
    else:examples[0].split='valid'
    with pytest.raises(ValueError):summarize_signal(audit,examples)


def sequence_fixture():
    audit, examples = fixture()
    audit['per_response'][0]['pair_grpo'] = 0.
    audit['per_response'][1]['pair_grpo'] = -2.
    records = [dict(example_id=r['example_id'], replicate=r['replicate'],
                    token_ids=[1, 2], log_probs=[-.3, -.5]) for r in audit['per_response']]
    return audit, examples, records


def test_identical_sequences_cancel_but_distinct_prompts_do_not_merge():
    audit, examples, records = sequence_fixture()
    result = summarize_sequence_coefficients(audit, examples, records)
    assert result['overall']['sequence_buckets'] == 3
    assert result['overall']['standard']['cancelled_material_buckets'] == 1
    assert result['overall']['standard']['material_buckets'] == 0
    assert result['overall']['center_only']['cancelled_material_buckets'] == 2


def test_distinct_token_sequences_preserve_signal_and_report_old_logprob_spread():
    audit, examples, records = sequence_fixture()
    records[1]['token_ids'] = [1, 3]
    records[3]['log_probs'] = [-.4, -.5]
    result = summarize_sequence_coefficients(audit, examples, records)
    assert result['overall']['standard']['material_buckets'] == 2
    assert result['overall']['standard']['material_prompts'] == 1
    assert result['overall']['max_old_logprob_spread'] == pytest.approx(.1)


@pytest.mark.parametrize('change', ['order', 'length', 'nan', 'empty'])
def test_sequence_projection_rejects_invalid_records(change):
    audit, examples, records = sequence_fixture()
    if change == 'order': records.reverse()
    elif change == 'length': records[0]['log_probs'].pop()
    elif change == 'nan': records[0]['log_probs'][0] = float('nan')
    else: records[0]['token_ids'] = []
    with pytest.raises(ValueError): summarize_sequence_coefficients(audit, examples, records)
