import copy
import hashlib
import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
from infer_fsc_context_transfer import checked_prediction,validate_source
from multimodal_jev.slurp_training import content_hash


def fixture():
    receipt=dict(positions=[list(range(31))],tensors=dict(input_ids={'sha256':'input'},positions={'sha256':'positions'}))
    identity=dict(policy_id='policy',positions=receipt['positions'][0],inputs={'input_ids':{'sha256':'input'}})
    pred=dict(policy_id='policy',request_identity=identity,request_id=content_hash(identity),prob=[.1]*31)
    return receipt,pred


def test_engine_binding():
    receipt,pred=fixture()
    assert checked_prediction({'id':'audio'},receipt,pred,'policy')['id']=='audio'


@pytest.mark.parametrize('fault',['policy','inputs','positions','digest','count','nan','range'])
def test_wrong_engine_output(fault):
    receipt,pred=fixture();pred=copy.deepcopy(pred)
    if fault=='policy':pred['policy_id']='stale'
    elif fault=='inputs':pred['request_identity']['inputs']={}
    elif fault=='positions':pred['request_identity']['positions']=[1]*31
    elif fault=='digest':pred['request_id']='other'
    elif fault=='count':pred['prob'].pop()
    elif fault=='nan':pred['prob'][0]=float('nan')
    else:pred['prob'][0]=1.01
    with pytest.raises(ValueError):checked_prediction({'id':'audio'},receipt,pred,'policy')


def test_source_only_preselection(tmp_path):
    path=tmp_path/'source.json'
    def config(complete=True):
        path.write_text(json.dumps(dict(complete=complete,initial_policy_id='initial',
                          selected_checkpoint=dict(policy_id='selected',step=64))))
        return dict(source_selection=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
                    source_role='selected',policy_id='selected')
    c=config();validate_source(c)
    c['policy_id']='target-best'
    with pytest.raises(ValueError):validate_source(c)
    c=config(False)
    with pytest.raises(ValueError):validate_source(c)
    c=config();path.write_text('{}')
    with pytest.raises(ValueError):validate_source(c)
