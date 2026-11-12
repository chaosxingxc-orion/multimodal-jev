import copy
import pytest
from multimodal_jev.intent_learning_comparison import compare


def fixture():
    rows=[dict(id='a',gold=['yes'],prediction=['no'],correct=False,gold_probability=[.1]),
          dict(id='b',gold=['no'],prediction=['no'],correct=True,gold_probability=[.8])]
    before=dict(role='learning',base_id='base',adapter_id='old',reports={a:dict(admission={'fixed':True},metrics={'records':copy.deepcopy(rows)}) for a in ('D','Q')})
    after=copy.deepcopy(before);after['adapter_id']='new'
    for arm in ('D','Q'):
        a,b=after['reports'][arm]['metrics']['records']
        a.update(prediction=['yes'],correct=True,gold_probability=[.7])
        b.update(prediction=['yes'],correct=False,gold_probability=[.4])
    return before,after


def test_offsets_do_not_hide_update_improvement_and_other_source_regression():
    a,b=fixture();result=compare(a,b,['a'])['reports']['D']
    assert result['all_learning']['delta']['final_accuracy']==0
    assert result['updated_this_step']['wrong_to_right']==1
    assert result['other_learning']['right_to_wrong']==1
    assert result['other_learning']['delta']['final_nll']>0


@pytest.mark.parametrize('bad',['missing','gold','selected','base'])
def test_rejects_nonpaired_evidence(bad):
    a,b=fixture();selected=['a']
    if bad=='missing':b['reports']['D']['metrics']['records'].pop()
    if bad=='gold':b['reports']['Q']['metrics']['records'][0]['gold']=['changed']
    if bad=='selected':selected=['unseen']
    if bad=='base':b['base_id']='different'
    with pytest.raises(ValueError):compare(a,b,selected)
