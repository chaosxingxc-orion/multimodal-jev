import copy
import pytest
from tools.reports.context_infonce_formal_resume import derive
from tools.reports.audit_context_infonce_training import expected_contract
from test_context_infonce_training_audit import fixture


def example():
    c,_=fixture();c.update(stop_update=160,start_update=0,source_root='/old/source',schedule='/old/source/schedule',parent_schedule='/old/source/parent-schedule',pins={'/old/source/a':'a','/model/base':'b'},resume_root=None,experiment_kind='formal_infonce_source')
    m=dict(contract=expected_contract(c),updates=32,rank_shards={f'rank-{r}/checkpoint-0032.pt':str(r) for r in range(8)})
    return c,m


def test_resume_preserves_optimizer_contract_and_schedule():
    c,m=example();old=copy.deepcopy(c);r=derive(c,'/old','/new',m,'commit')
    assert expected_contract(r)==expected_contract(c)
    assert r['start_update']==32 and r['stop_update']==160
    assert r['resume_root']=='/old/run'
    assert r['pins']['/old/run/checkpoint-0032-committed.json']=='commit'
    assert r['pins']['/new/source/a']=='a' and c==old


@pytest.mark.parametrize('fault',['contract','cursor','shards'])
def test_resume_rejects_incompatible_checkpoint(fault):
    c,m=example()
    if fault=='contract':m['contract']['microbatch']=16
    elif fault=='cursor':m['updates']=160
    else:m['rank_shards'].pop('rank-0/checkpoint-0032.pt')
    with pytest.raises(ValueError):derive(c,'/old','/new',m,'commit')
