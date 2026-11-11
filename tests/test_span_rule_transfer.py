from copy import deepcopy
from collections import Counter
import itertools
import pytest
from test_span_rule_dataset import bundle
from multimodal_jev.span_rule_dataset import build_rule_controls
from multimodal_jev.span_rule_balance import add_complementary_rules
from multimodal_jev.span_rule_transfer import semantic_rule_group,EXCLUSION_GROUP,make_transfer_protocol

A={'op':'fact','key':'a'};B={'op':'fact','key':'b'}

def populations():
    learning=add_complementary_rules(build_rule_controls([bundle(v) for v in itertools.product((False,True),repeat=3)])[0])
    selection=deepcopy(learning)
    for r in selection:
        r['id']='reserved:'+r['id'];r['split']='selection';r['optimization_allowed']=False
        for k in ('source_group','audio_sha256','recording_id'):
            r['provenance'][k]='reserved:'+r['provenance'][k]
    return learning,selection


def test_semantic_equivalence_groups_close_syntax_complement_and_permutation():
    variants=[{'op':'except','base':A,'exception':B},
              {'op':'all','args':[A,{'op':'not','arg':B}]},
              {'op':'except','base':B,'exception':A},
              {'op':'any','args':[{'op':'not','arg':A},B]}]
    assert {semantic_rule_group(v) for v in variants}=={EXCLUSION_GROUP}
    both={'op':'all','args':[A,B]}
    demorgan={'op':'not','arg':{'op':'any','args':[{'op':'not','arg':A},{'op':'not','arg':B}]}}
    assert semantic_rule_group(both)==semantic_rule_group(demorgan)
    assert semantic_rule_group(both)!=EXCLUSION_GROUP
    assert semantic_rule_group({'op':'any','args':[A,B]})!=EXCLUSION_GROUP


def test_source_task_cells_and_all_seed_batches_remain_disjoint_and_balanced():
    learning,selection=populations();p=make_transfer_protocol(learning,selection,updates=12)
    by_id={r['id']:r for r in learning}
    for batches in p['schedules'].values():
        assert len(batches)==12
        assert set(Counter(b['rule_pair_id'] for b in batches).values())=={2}
        for batch in batches:
            for arm,ids in batch['arms'].items():
                rows=[by_id[k] for k in ids]
                assert all(semantic_rule_group(r['provenance']['rule'])!=EXCLUSION_GROUP for r in rows)
                assert all(sum(r['targets'][j] for r in rows)==4 for j in range(len(rows[0]['targets'])))
    cells=p['evaluation_cells'];rosters=[set(c['request_ids']) for c in cells.values()]
    assert len(set.union(*rosters))==len(learning)+len(selection)
    assert all(not a&b for a,b in itertools.combinations(rosters,2))
    assert p==make_transfer_protocol(learning,selection,updates=12)


@pytest.mark.parametrize('fault',['overlap','selection_training','gold','numeric_gold','missing_arm','missing_holdout','too_many','bad_seed'])
def test_reject_invalid_transfer_plan(fault):
    learning,selection=populations();kwargs=dict(updates=12)
    if fault=='overlap':selection[0]['provenance']['audio_sha256']=learning[0]['provenance']['audio_sha256']
    elif fault=='selection_training':selection[0]['optimization_allowed']=True
    elif fault=='gold':learning[0]['targets'][-1]=not learning[0]['targets'][-1]
    elif fault=='numeric_gold':learning[0]['targets'][-1]=int(learning[0]['targets'][-1])
    elif fault=='missing_arm':selection.pop()
    elif fault=='missing_holdout':learning=[r for r in learning if semantic_rule_group(r['provenance']['rule'])!=EXCLUSION_GROUP]
    elif fault=='too_many':kwargs['updates']=100
    elif fault=='bad_seed':kwargs['seeds']=(17,17)
    with pytest.raises(ValueError):make_transfer_protocol(learning,selection,**kwargs)
