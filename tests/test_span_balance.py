from itertools import product
import pytest
from multimodal_jev.span_balance import balanced_binary_batches


def test_joint_balance_and_every_batch_marginal_balance_without_repeats():
    bundles=[dict(id=f'{sig}-{j}',targets=list(sig),supervised=[True]*3)
             for sig in product([False,True],repeat=3) for j in range(5+sum(sig))]
    bundles.append(dict(id='missing',targets=[False,None,True],supervised=[True,False,True]))
    batches,r=balanced_binary_batches(bundles,17)
    assert r['selected_recordings']==40 and r['updates']==20
    used=[i for batch in batches for i in batch]
    assert len(set(used))==40
    assert all(all(sum(bundles[i]['targets'][j] for i in batch)==1 for j in range(3)) for batch in batches)
    assert all(bundles[i]['id']!='missing' for i in used)
    again,_=balanced_binary_batches(bundles[::-1],17)
    assert [[bundles[::-1][i]['id'] for i in b] for b in again]==r['batch_recording_ids']


def test_missing_signature_rejected():
    with pytest.raises(ValueError,match='all joint'):
        balanced_binary_batches([dict(id='a',targets=[False],supervised=[True])],17)


def test_explicit_cursor_has_no_overlap_and_no_silent_wrap():
    from multimodal_jev.span_balance import select_balanced_step
    bundles=[dict(id=f'{sig}-{i}',targets=list(sig),supervised=[True]*3)
             for sig in product([False,True],repeat=3) for i in range(2)]
    first,_,a=select_balanced_step(bundles,17,0)
    second,_,b=select_balanced_step(bundles,17,1)
    assert not set(x['id'] for x in first)&set(x['id'] for x in second)
    assert a['schedule_sha256']==b['schedule_sha256'] and a['next_batch_index']==b['batch_index']
    for index in (-1,8,True):
        with pytest.raises(ValueError,match='batch index'):
            select_balanced_step(bundles,17,index)
