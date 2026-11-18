import json
from pathlib import Path
import pytest
from tools import collect_training_artifacts as module


def fixture(tmp_path):
    root = tmp_path/'collection'; root.mkdir()
    files = {}
    for name in ('config.json', 'service-config.json', 'PREPARED.json'):
        (root/name).write_text('{}'); files[name] = module.sha(root/name)
    archive = root.with_suffix('.tar'); archive.write_bytes(b'fixture archive')
    launch = dict(container_id='a'*64, image='sha256:'+'b'*64, remote_root='/data/run',
        config_sha256=files['config.json'], service_config_sha256=files['service-config.json'],
        manifest_sha256=files['PREPARED.json'])
    receipt = dict(launch, status='transport_complete_independent_audit_required',
        state=dict(Status='exited', ExitCode=0), files=files, bad_pins=[], bad_package=[],
        launch_binding_mismatches=[], archive_sha256=module.sha(archive))
    root.with_suffix('.transport.json').write_text(json.dumps(receipt))
    lp=tmp_path/'launch.json'; lp.write_text(json.dumps(launch))
    config=dict(schema='configured-training-collection-v1', mode='adopt', host='npu',
        launch=str(lp), launch_sha256=module.sha(lp), collection=str(root),
        output_directory=str(tmp_path/'publication'), lock_path=str(tmp_path/'shared.lock'))
    tp=tmp_path/'config.json';tp.write_text(json.dumps(config))
    local=tmp_path/'local.json';local.write_text(json.dumps(dict(schema='jev-local-macros-v1',macros={})))
    return root, launch, receipt, config, tp, local


def test_adopt_idempotent_without_remote_and_rechecks_artifacts(tmp_path, monkeypatch):
    root, _, _, _, template, local = fixture(tmp_path)
    monkeypatch.setattr(module,'check_paths',lambda *a:[])
    monkeypatch.setattr(module,'collect',lambda *a,**k:pytest.fail('no remote in adopt'))
    assert module.run(template,local)['status']=='complete'
    assert module.run(template,local)['status']=='already_complete'
    (root/'config.json').write_text('changed')
    with pytest.raises(ValueError,match='identity changed'):module.run(template,local)


@pytest.mark.parametrize('fault',['partial','live','identity','escape','archive','pins','launch_binding'])
def test_rejects_incomplete_or_changed_collection(tmp_path,fault):
    root,launch,receipt,*_=fixture(tmp_path)
    if fault=='partial':receipt['status']='transport_pending_not_audited'
    elif fault=='live':receipt['state']['Status']='running'
    elif fault=='identity':receipt['container_id']='c'*64
    elif fault=='escape':receipt['files']['../outside']='0'*64
    elif fault=='archive':root.with_suffix('.tar').write_bytes(b'changed')
    elif fault=='pins':receipt['bad_pins']=['model']
    elif fault=='launch_binding':launch['config_sha256']='f'*64
    root.with_suffix('.transport.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError):module.authenticate(launch,root)


def test_collect_calls_existing_collector_once_and_retains_failure(tmp_path,monkeypatch):
    root,launch,receipt,config,template,local=fixture(tmp_path)
    config.update(mode='collect',collection=str(tmp_path/'new-collection'))
    template.write_text(json.dumps(config));calls=[]
    monkeypatch.setattr(module,'check_paths',lambda *a:[])
    def fail(l,p,host):
        calls.append((l,p,host));raise RuntimeError('transport interrupted')
    monkeypatch.setattr(module,'collect',fail)
    with pytest.raises(RuntimeError):module.run(template,local)
    with pytest.raises(ValueError,match='Partial'):module.run(template,local)
    assert len(calls)==1 and calls[0][0]==launch and calls[0][2]=='npu'
    assert not (Path(config['output_directory'])/'COMPLETE.json').exists()
