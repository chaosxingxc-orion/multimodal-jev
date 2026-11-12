import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import report_intent_utility_curves as module
from multimodal_jev.baseline import atomic_json,file_hash
from test_intent_utility_curves import fixture as protocol_fixture


def fixture(tmp_path,monkeypatch):
    protocol,pilot=protocol_fixture()
    common=dict(model_path='/model',arm='D',lora_rank=8,lora_alpha=16,
        lora_execution_dtype='bfloat16',runtime_profile='fixed')
    for arm in protocol['arms']:
        prior=None
        for index,step in enumerate(arm['steps']):
            for key in ('training_root','suite_root'):step[key]=str(tmp_path/step[key])
            train,suite=Path(step['training_root']),Path(step['suite_root'])
            cfg=dict(common,**{k:protocol[k] for k in ('pilot','manifest_sha256','seed','lr','global_requests')},
                step=index,objective=arm['objective'],pins={},initial_adapter=protocol['initial_adapter'],
                acceptance_root=protocol['initial_gate'])
            if prior:cfg.update(previous_root=prior['training_root'],previous_suite=prior['suite_root'],
                initial_adapter=prior['training_root']+'/run/adapter',acceptance_root=prior['suite_root'])
            for root in (train,suite):
                root.mkdir(parents=True);atomic_json(root/'config.json',cfg if root==train else {'training_root':str(train)})
                atomic_json(root/'source-manifest.json',{})
                atomic_json(root/'launch.json',dict(image='image',config_sha256=file_hash(root/'config.json'),
                    source_manifest_sha256=file_hash(root/'source-manifest.json')))
            (train/'run').mkdir();atomic_json(train/'run/selection.json',dict(recording_ids=protocol['schedule'][index]))
            prior=step
    pilot['arms']=[dict(a,steps=copy.deepcopy(a['steps'][:2])) for a in protocol['arms']]
    pilot_path=tmp_path/'pilot.json';atomic_json(pilot_path,pilot)
    protocol.update(pilot_protocol=str(pilot_path),pilot_protocol_sha256=file_hash(pilot_path))
    path=tmp_path/'protocol.json';atomic_json(path,protocol);calls=[]
    def report(training,suite):
        calls.append(str(training));cfg=module.read(training/'config.json')
        return dict(files={},groups={'all_learning':{'recordings':89}},training_trace={'updates':cfg['step']+1})
    monkeypatch.setattr(module,'update_report',report)
    return path,protocol,calls


def change_config(root,**changes):
    p=Path(root);cfg=module.read(p/'config.json');cfg.update(changes);atomic_json(p/'config.json',cfg)
    launch=module.read(p/'launch.json');launch['config_sha256']=file_hash(p/'config.json');atomic_json(p/'launch.json',launch)


def test_all_three_curves_and_every_update_are_authenticated(tmp_path,monkeypatch):
    path,p,calls=fixture(tmp_path,monkeypatch);r=module.run(path,file_hash(path))
    assert r['complete'] and len(calls)==36
    assert set(r['curves'])=={a['name'] for a in p['arms']}
    for points in r['curves'].values():
        assert [v['update'] for v in points]==list(range(1,13))
        assert points[-1]['source_uses']==96


@pytest.mark.parametrize('bad',['missing','objective','optimizer','schedule','lineage','source','pilot_hash','protocol_hash'])
def test_incomplete_or_changed_protocol_never_scores(tmp_path,monkeypatch,bad):
    path,p,calls=fixture(tmp_path,monkeypatch);digest=file_hash(path)
    step=p['arms'][-1]['steps'][-1];root=Path(step['training_root'])
    if bad=='missing':(root/'launch.json').unlink()
    elif bad=='objective':change_config(root,objective={'name':'correctness-v1'})
    elif bad=='optimizer':change_config(root,adam_betas=[0.,.999])
    elif bad=='schedule':atomic_json(root/'run/selection.json',{'recording_ids':['other']})
    elif bad=='lineage':change_config(root,previous_root='/wrong',initial_adapter='/wrong/run/adapter')
    elif bad=='source':
        atomic_json(root/'source-manifest.json',{'different':'source'})
        launch=module.read(root/'launch.json');launch['source_manifest_sha256']=file_hash(root/'source-manifest.json')
        atomic_json(root/'launch.json',launch)
    elif bad=='pilot_hash':atomic_json(Path(p['pilot_protocol']),{})
    else:digest='wrong'
    with pytest.raises(ValueError):module.run(path,digest)
    assert calls==[]


def test_actual_gate_failure_cannot_publish_complete_result(tmp_path,monkeypatch):
    path,p,calls=fixture(tmp_path,monkeypatch)
    def failed(*args):raise ValueError('actual container failed')
    monkeypatch.setattr(module,'update_report',failed)
    with pytest.raises(ValueError,match='actual container'):module.run(path,file_hash(path))
