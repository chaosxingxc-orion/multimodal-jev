import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from multimodal_jev.baseline import file_hash


@pytest.fixture
def module(monkeypatch):
    path=Path(__file__).resolve().parents[1]/'tools/jev/queue_span_pair_pilot.py'
    monkeypatch.syspath_prepend(str(path.parent));spec=importlib.util.spec_from_file_location('pair_queue',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


@pytest.mark.parametrize('fault',[None,'reserved','privileged','host_process','transport','stderr','root_busy','root_denied'])
def test_card_ownership_check_fails_closed(module,monkeypatch,fault):
    def output(argv,**kw):
        if argv[-2:]==['ps','-q']:return 'container\n'
        return json.dumps([dict(Id='container',HostConfig=dict(Privileged=fault=='privileged',
            Devices=[dict(PathOnHost='/dev/davinci5' if fault=='reserved' else '/dev/davinci6')]))])
    monkeypatch.setattr(module.subprocess,'check_output',output)
    def run(argv,**kw):
        if 'fuser' not in argv:
            return SimpleNamespace(returncode=1 if fault in ('root_busy','root_denied') else 0,
                stdout=json.dumps(dict(busy=[123] if fault=='root_busy' else [],denied=[123] if fault=='root_denied' else [])),stderr='')
        return SimpleNamespace(returncode=0 if fault=='host_process' else 255 if fault=='transport' else 1,
            stdout='123' if fault=='host_process' else '',stderr='error' if fault=='stderr' else '')
    monkeypatch.setattr(module.subprocess,'run',run)
    if fault:
        with pytest.raises(ValueError):module.check_card5()
    else:module.check_card5()


@pytest.mark.parametrize('fault',[None,'wait','pin','plan','create','start','busy'])
def test_exact_wait_and_receipt_before_start(module,tmp_path,monkeypatch,fault):
    root=tmp_path/'training';root.mkdir();output=tmp_path/'output';output.mkdir()
    preflight_root=tmp_path/'preflight';preflight_root.mkdir()
    (preflight_root/'result.json').write_text(json.dumps(dict(source_manifest_sha256='source')))
    evidence=tmp_path/'preflight.json';evidence.write_text(json.dumps(dict(passed=True,files={},root=str(preflight_root))))
    plan=dict(create_argv=['docker','create','name'],receipt_fields=dict(image='image',source_manifest_sha256='source'))
    plan_path=tmp_path/'plan.json';plan_path.write_text(json.dumps(plan))
    config=dict(pins={str(evidence):file_hash(evidence)},wait_container_id='original',training_root=str(root),device_lock=str(tmp_path/'lock'),
                preflight=str(evidence),launch_plan=str(plan_path),launch_receipt=str(tmp_path/'receipt.json'))
    if fault=='pin':evidence.write_text('{}')
    monkeypatch.setattr(module,'prepare',lambda root:{**plan,'changed':True} if fault=='plan' else plan)
    def card():
        if fault=='busy':raise ValueError('busy')
    monkeypatch.setattr(module,'check_card5',card)
    monkeypatch.setattr(module.subprocess,'check_output',lambda argv,**kw:json.dumps([dict(Id=argv[-1],Image='image',State=dict(Status='running'))]))
    calls=[];container='a'*64
    def run(argv,**kw):
        if 'wait' in argv:return SimpleNamespace(returncode=0,stdout='1' if fault=='wait' else '0',stderr='')
        calls.append(argv)
        if 'start' in argv:
            assert json.loads((root/'launch.json').read_text())['container_id']==container
            return SimpleNamespace(returncode=1 if fault=='start' else 0,stdout=container,stderr='')
        return SimpleNamespace(returncode=1 if fault=='create' else 0,stdout=container,stderr='')
    monkeypatch.setattr(module.subprocess,'run',run)
    if fault:
        with pytest.raises(ValueError):module.run(config,output)
        assert not (output/'dispatched.json').exists()
        if fault not in ('create','start'):assert not calls
    else:
        module.run(config,output)
        assert len(calls)==2 and (output/'dispatched.json').exists()
