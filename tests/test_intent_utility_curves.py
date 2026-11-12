import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from prepare_intent_utility_curves import validate_protocol
from prepare_intent_utility_pilot import OBJECTIVES


def fixture():
    common=dict(initial_adapter='adapter',initial_gate='gate',pilot='pilot',manifest_sha256='hash',seed=317,lr=1e-4,global_requests=8)
    arms=[dict(name=o['name'],objective=o,steps=[dict(training_root=f'{j}/t{i}',suite_root=f'{j}/s{i}') for i in range(12)]) for j,o in enumerate(OBJECTIVES)]
    batches=[[str((i*8+k)%89) for k in range(8)] for i in range(12)]
    p=dict(common,schema='intent-objective-curves-v1',initial_updates=2,total_updates=12,source_uses_per_arm=96,checkpoints=list(range(13)),arms=arms,schedule=batches)
    old=dict(common,schema='intent-objective-engineering-pilot-v1',updates_per_arm=2,
        arms=[dict(a,steps=copy.deepcopy(a['steps'][:2])) for a in arms],schedule=copy.deepcopy(batches[:2]))
    return p,old


def test_complete_matched_curves():validate_protocol(*fixture())


@pytest.mark.parametrize('bad',['arm','budget','checkpoint','initial','schedule','duplicate','objective','lr','sources'])
def test_rejects_selection_or_changed_factors(bad):
    p,old=fixture()
    if bad=='arm':p['arms'].pop()
    elif bad=='budget':p['total_updates']=13
    elif bad=='checkpoint':p['checkpoints'].remove(4)
    elif bad=='initial':p['arms'][0]['steps'][0]['training_root']='changed'
    elif bad=='schedule':p['schedule'][0].reverse()
    elif bad=='duplicate':p['arms'][0]['steps'][2]=p['arms'][0]['steps'][1]
    elif bad=='objective':p['arms'][1]['objective']={'name':'uniform-credit-v1','scale':9.}
    elif bad=='lr':p['lr']=2e-5
    else:p['schedule'][5][1]=p['schedule'][5][0]
    with pytest.raises(ValueError):validate_protocol(p,old)
