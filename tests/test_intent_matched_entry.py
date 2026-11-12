import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import compare_intent_matched_pilots as module
from multimodal_jev.baseline import atomic_json,file_hash
from test_intent_matched_effects import score


def fixture(tmp_path,monkeypatch):
    initial=tmp_path/'initial';initial.mkdir()
    atomic_json(initial/'launch.json',dict(image='image',container_id='initial'))
    protocol=tmp_path/'protocol.json'
    atomic_json(protocol,dict(total_updates=1,seed=317,lr=1e-4,global_requests_per_update=8,schedule=[['a']]))
    c=dict(protocol=str(protocol),initial_gate=str(initial),image='image',arms={},
        pins={str(p):file_hash(p) for p in (protocol,initial/'launch.json')})
    cfg=dict(arm='D',step=0,seed=317,lr=1e-4,global_requests=8,acceptance_root=str(initial),initial_adapter='/initial',pins={})
    for arm in ('D','Q','P'):
        train=tmp_path/arm;suite=tmp_path/(arm+'-suite');train.mkdir();suite.mkdir();(train/'run').mkdir()
        atomic_json(train/'config.json',dict(cfg,arm=arm))
        for root in (train,suite):atomic_json(root/'launch.json',dict(image='image'))
        atomic_json(train/'run/selection.json',dict(recording_ids=['a']))
        c['arms'][arm]=[dict(training_root=str(train),suite_root=str(suite))]
    monkeypatch.setattr(module,'completed',lambda *a:True)
    calls=[]
    def checked(train,suite):
        calls.append(train.name)
        return dict(full_before=score(False,'initial'),full_after=score(train.name=='P',train.name),pins={})
    monkeypatch.setattr(module,'score_update',checked)
    return c,calls


def test_authenticates_each_training_arm(tmp_path,monkeypatch):
    c,calls=fixture(tmp_path,monkeypatch);result=module.run(c)
    assert calls==['D','Q','P'] and result['complete']
    assert result['curve'][-1]['effects']['process_credit']['final_difference']['final_accuracy']==1


@pytest.mark.parametrize('bad',['incomplete','wrong_arm','optimizer','schedule','image','initial_exit','scoring_failure'])
def test_rejects_spliced_or_unverified_runs(tmp_path,monkeypatch,bad):
    c,calls=fixture(tmp_path,monkeypatch)
    train=Path(c['arms']['Q'][0]['training_root'])
    if bad=='incomplete':c['arms']['P']=[]
    elif bad in ('wrong_arm','optimizer'):
        p=train/'config.json';v=module.read(p);v['arm' if bad=='wrong_arm' else 'adam_betas']='D' if bad=='wrong_arm' else [0.,.999];atomic_json(p,v)
    elif bad=='schedule':atomic_json(train/'run/selection.json',dict(recording_ids=['other']))
    elif bad=='image':atomic_json(train/'launch.json',dict(image='other'))
    elif bad=='initial_exit':monkeypatch.setattr(module,'completed',lambda *a:False)
    elif bad=='scoring_failure':
        def fail(*a):raise ValueError('actual gate failed')
        monkeypatch.setattr(module,'score_update',fail)
    with pytest.raises(ValueError):module.run(c)
