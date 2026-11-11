from copy import deepcopy
import itertools
import pytest
from test_span_rule_dataset import bundle
from multimodal_jev.span_rule_dataset import build_rule_controls
from multimodal_jev.span_rule_balance import add_complementary_rules,balanced_rule_schedule


def fixture():
    rows,_=build_rule_controls([bundle(v) for v in itertools.product((False,True),repeat=3)])
    return add_complementary_rules(rows)


def test_every_batch_balances_each_span_and_covers_each_rule_truth_table():
    rows=fixture();by_id={r['id']:r for r in rows};schedule=balanced_rule_schedule(rows,17)
    assert len(schedule['batches'])==18
    seen={arm:set() for arm in ('final_only','structured','reordered')}
    for batch in schedule['batches']:
        assert len(set(batch['source_groups']))==4
        for arm,ids in batch['arms'].items():
            assert not seen[arm].intersection(ids);seen[arm].update(ids)
            actual=[by_id[i] for i in ids]
            assert len(actual)==8
            assert all(sum(r['targets'][j] for r in actual)==4 for j in range(len(actual[0]['targets'])))
            for polarity in ('positive','negative'):
                selected=[r for r in actual if r['provenance']['rule_polarity']==polarity]
                assert {tuple(r['provenance']['dependency_targets']) for r in selected}==set(itertools.product((False,True),repeat=2))
        # Arm comparison sees exactly the same recording/rule order.
        assert [[(by_id[i]['provenance']['recording_id'],by_id[i]['provenance']['rule_polarity']) for i in ids]
                for ids in batch['arms'].values()].count([(by_id[i]['provenance']['recording_id'],by_id[i]['provenance']['rule_polarity']) for i in batch['arms']['structured']])==3
    assert schedule==balanced_rule_schedule(rows,17)
    assert schedule!=balanced_rule_schedule(rows,18)


def test_complement_is_visible_and_does_not_change_intermediate_truth():
    rows=fixture()
    for positive,negative in zip(rows[::2],rows[1::2],strict=True):
        assert positive['request']['questions'][:-1]==negative['request']['questions'][:-1]
        assert positive['request']['questions'][-1]['schema']['question'] in negative['request']['questions'][-1]['schema']['question']
        assert negative['targets'][-1] is not positive['targets'][-1]
        assert negative['targets'][:-1]==positive['targets'][:-1]


@pytest.mark.parametrize('fault',['missing_polarity','missing_arm','duplicate','wrong_target','missing_signature'])
def test_reject_incomplete_or_corrupt_sampling_pool(fault):
    rows=fixture()
    if fault=='missing_polarity':rows.pop()
    elif fault=='missing_arm':rows=rows[:-2]
    elif fault=='duplicate':rows.append(deepcopy(rows[0]))
    elif fault=='wrong_target':rows[1]['targets'][-1]=rows[0]['targets'][-1]
    else:rows=[r for r in rows if r['provenance']['dependency_targets']!=[True,True]]
    with pytest.raises(ValueError):balanced_rule_schedule(rows,17)
