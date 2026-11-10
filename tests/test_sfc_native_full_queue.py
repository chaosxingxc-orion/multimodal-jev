import json
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from test_sfc_jev import load_tool


def queue(tmp_path):
    predecessor=tmp_path/'jev';(predecessor/'score').mkdir(parents=True)
    (predecessor/'score/metrics.json').write_text('{}')
    (predecessor/'acceptance.json').write_bytes(canonical(dict(bundles=6,decisions=78,source_groups=4,weight_updates=0,
        metrics_sha256=file_hash(predecessor/'score/metrics.json'))))
    root=tmp_path/'queue';root.mkdir()
    plan=dict(dependency={'root':str(predecessor)},dependency_seconds=110000,splits=['test_id','test_ood'],
              official_scorer='/scorer',official_scorer_sha256='hash')
    (root/'queue-plan.json').write_bytes(canonical(plan))
    return root


def test_wait_then_engine_acceptance_then_complete_id_and_ood(tmp_path,monkeypatch):
    m=load_tool('watch_sfc_native_full');root=queue(tmp_path);events=[];ready=iter([False,True])
    monkeypatch.setattr(m,'check_pins',lambda *a:None)
    monkeypatch.setattr(m,'observe_dependency',lambda _:next(ready))
    monkeypatch.setattr(m.time,'sleep',lambda _:events.append('wait'))
    monkeypatch.setattr(m,'accept_jev_probe',lambda _:events.append('accept78'))
    def prepare(root,plan,split):
        child=root/split;child.mkdir();events.append('prepare-'+split);return child,{}
    monkeypatch.setattr(m,'prepare_split',prepare)
    monkeypatch.setattr(m,'dispatch_once',lambda child,_:events.append('dispatch-'+child.name))
    monkeypatch.setattr(m,'score',lambda child,*a:events.append('score-'+child.name))
    m.execute(root)
    assert events==['wait','accept78','prepare-test_id','dispatch-test_id','score-test_id',
                    'prepare-test_ood','dispatch-test_ood','score-test_ood']
    with pytest.raises(FileExistsError):m.execute(root)


def test_transient_observation_error_does_not_dispatch_or_restart(tmp_path,monkeypatch):
    m=load_tool('watch_sfc_native_full');root=queue(tmp_path);calls=[]
    monkeypatch.setattr(m,'check_pins',lambda *a:None)
    def observe(_):
        calls.append('poll')
        if len(calls)==1:raise subprocess.TimeoutExpired('ssh',20)
        raise RuntimeError('predecessor failed')
    monkeypatch.setattr(m,'observe_dependency',observe)
    monkeypatch.setattr(m.time,'sleep',lambda _:None)
    monkeypatch.setattr(m,'prepare_split',lambda *a:pytest.fail('no preparation'))
    with pytest.raises(RuntimeError,match='predecessor'):m.execute(root)
    assert calls==['poll','poll']


def prep(tmp_path):
    (tmp_path/'config.json').write_text('{}');(tmp_path/'source').mkdir()
    p=dict(config_sha256=file_hash(tmp_path/'config.json'),source_files={},image='image',
           argv=['docker','run','--device','/dev/davinci6:/dev/davinci6','--device','/dev/davinci7:/dev/davinci7'])
    (tmp_path/'preparation.json').write_bytes(canonical(p));return p


def test_ambiguous_dispatch_remains_nonreplayable(tmp_path,monkeypatch):
    m=load_tool('watch_sfc_native_full');p=prep(tmp_path);calls=[]
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:'')
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:calls.append('dispatch') or NS(returncode=0,stdout='unknown',stderr=''))
    with pytest.raises(RuntimeError,match='ambiguous'):m.dispatch_once(tmp_path,p)
    with pytest.raises(FileExistsError):m.dispatch_once(tmp_path,p)
    assert calls==['dispatch']


def test_busy_cards_prevent_dispatch(tmp_path,monkeypatch):
    m=load_tool('watch_sfc_native_full');p=prep(tmp_path)
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:'cid')
    monkeypatch.setattr(m,'inspect',lambda _:dict(HostConfig=dict(Devices=[dict(PathOnHost='/dev/davinci6')])))
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:pytest.fail('must not dispatch'))
    with pytest.raises(RuntimeError,match='occupied'):m.dispatch_once(tmp_path,p)
    assert not (tmp_path/'dispatch-intent.json').exists()


def test_exact_dispatch_waits_same_container_and_preserves_launch(tmp_path,monkeypatch):
    m=load_tool('watch_sfc_native_full');p=prep(tmp_path);cid='a'*64;seen=[]
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:'')
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:NS(returncode=0,stdout=cid+'\n',stderr=''))
    monkeypatch.setattr(m,'wait',lambda root,ident,image,deadline:seen.append((ident,image)) or dict(Status='exited',ExitCode=0))
    m.dispatch_once(tmp_path,p)
    assert seen==[(cid,'image')]
    assert json.loads((tmp_path/'launch.json').read_text())['config_sha256']==p['config_sha256']


def test_preparation_pins_full_splits_native_model_and_readonly_source(tmp_path,monkeypatch):
    m=load_tool('watch_sfc_native_full');root=tmp_path/'queue';root.mkdir()
    native=tmp_path/'native';native.mkdir();release=tmp_path/'release';release.mkdir()
    (native/'config.json').write_bytes(canonical(dict(audio_model='/model',versions={'vllm':'pin'},asset_audit='/audit')))
    (native/'launch.json').write_text('{"container_id":"cid"}')
    (native/'acceptance.json').write_text('{}')
    (release/'MANIFEST.json').write_bytes(canonical(dict(splits={'test_id':dict(requests=2590,requests_sha256='a'*64)})))
    (root/'source').mkdir();(root/'source/runner.py').write_text('source')
    monkeypatch.setattr(m,'SOURCE_FILES',('runner.py',))
    monkeypatch.setattr(m,'select_inputs',lambda _:None);monkeypatch.setattr(m,'check_probe',lambda _:None)
    monkeypatch.setattr(m,'inspect',lambda _:dict(Id='cid',Image='image',State=dict(Status='exited',ExitCode=0)))
    plan=dict(splits=['test_id','test_ood'],native_probe=str(native),release=str(release),image='image',source_commit='commit')
    child,p=m.prepare_split(root,plan,'test_id');cfg=json.loads((child/'config.json').read_text())
    assert cfg['max_new_tokens']==1024 and cfg['split']=='test_id' and p['requests']==2590
    assert f'{native}:{native}:ro' in p['argv'] and f'{root}/source:{root}/source:ro' in p['argv']
    assert (child/'source').resolve()==root/'source'
    with pytest.raises(FileExistsError):m.prepare_split(root,plan,'test_id')
