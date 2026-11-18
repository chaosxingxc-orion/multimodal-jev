from copy import deepcopy
from pathlib import Path
import runpy

import pytest

select=runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/jev/rescore_slurp_clean_native.py'))['clean_population']


def fixture():
    req=[{'id':'slurp:a'},{'id':'slurp:b'}]
    target={'scenario':'email','action':'sendemail','entities':[{'type':'person','filler':'amy'}]}
    gold=[dict(id=r['id'],target=deepcopy(target),metadata={'file':r['id'][6:],'slurp_id':i+1}) for i,r in enumerate(req)]
    metadata=[dict(slurp_id=1,scenario='email',action='sendemail',tokens=[dict(id=0,surface='Amy')],
                   entities=[dict(type='person',span=[0])],recordings=[{'file':'a'}])]
    return req,gold,metadata,['slurp:b']


def test_subset_preserves_original_identity_and_target():
    args=fixture();r,t=select(*args)
    assert r==args[0][:1] and t==args[1][:1]


@pytest.mark.parametrize('bad',['target','source','missing_exclusion','unknown_exclusion','duplicate_recording','duplicate_exclusion'])
def test_population_or_label_drift_rejected(bad):
    args=fixture()
    if bad=='target':args[2][0]['action']='query'
    if bad=='source':args[2][0]['slurp_id']=99
    if bad=='missing_exclusion':args[3].clear()
    if bad=='unknown_exclusion':args[3].append('slurp:unknown')
    if bad=='duplicate_recording':args[2][0]['recordings'].append({'file':'a'})
    if bad=='duplicate_exclusion':args[3].append('slurp:b')
    with pytest.raises(ValueError):select(*args)
