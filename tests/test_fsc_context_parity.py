import copy
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
from probe_fsc_context_parity import panel_indices,compare


def report():
    return dict(policy_id='policy',indices=[0],input_manifest_sha256='inputs',
                records=[dict(id='audio',request_identity={'policy_id':'policy'},prob=[.5]*31)])


def test_exact_and_visible_probability_errors():
    a=report();b=copy.deepcopy(a)
    assert compare(a,b)['exact']
    b['records'][0]['prob'][0]=.49
    result=compare(a,b)
    assert not result['exact'] and result['threshold_flips']==1
    assert result['probability_max_abs']==pytest.approx(.01)


@pytest.mark.parametrize('fault',['policy','indices','identity','nan','missing','partial'])
def test_incomparable_backend_records_rejected(fault):
    a=report();b=copy.deepcopy(a)
    if fault=='policy':b['policy_id']='other'
    elif fault=='indices':b['indices']=[1]
    elif fault=='identity':b['records'][0]['request_identity']={}
    elif fault=='nan':b['records'][0]['prob'][0]=float('nan')
    elif fault=='partial':a['indices']=b['indices']=[0,1]
    else:b['records'][0]['prob'].pop()
    with pytest.raises(ValueError):compare(a,b)


def test_label_free_panel_keeps_extremes_and_each_speaker():
    rows=[dict(id=str(i),speaker_id=str(i%10)) for i in range(100)]
    receipts=[dict(tokens=200+i) for i in range(100)]
    chosen=panel_indices(rows,receipts)
    assert set(range(16))<=set(chosen)
    assert set(range(84,100))<=set(chosen)
    assert {rows[i]['speaker_id'] for i in chosen}=={str(i) for i in range(10)}
    assert chosen==sorted(set(chosen))
