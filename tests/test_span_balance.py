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
