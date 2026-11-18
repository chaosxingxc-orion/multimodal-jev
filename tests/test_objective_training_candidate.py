import json
from pathlib import Path

import pytest

from tools.reports import prepare_objective_training_candidate as module
from tools.reports.context_objective_bootstrap import MODULES,OBJECTIVE

ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(module.encode(value))


def setup(tmp_path,monkeypatch):
    parent=tmp_path/'parent';snapshot=tmp_path/'cpu-source';parent.mkdir();snapshot.mkdir()
    runtime={'forward.py':module.sha(b'pass\n')}
    (parent/'source').mkdir();(parent/'source/forward.py').write_bytes(b'pass\n')
    write(parent/'RUNTIME-SOURCES.json',runtime)
    config=dict(experiment_kind='engineering_prefix',start_update=0,stop_update=2,
        source_root='/data/parent/source',dual_runtime_manifest='/data/parent/RUNTIME-SOURCES.json',
        dual_runtime_identity=dict(source_manifest_sha256=module.sha((parent/'RUNTIME-SOURCES.json').read_bytes())),
        reference_dataset=dict(root='/data/reference',manifest_sha256='ref'),pins={'/shared/model':'modelhash'})
    write(parent/'config.json',config)
    write(parent/'service-a.json',dict(source_root='/data/parent/source',pins={}))
    (parent/'run.py').write_text("actor_command=[str(r/'source/tools/reports/context_cached_reference_trainer.py')]\n")
    m=dict(remote_root='/data/parent',image='fixed',
        files={str(p.relative_to(parent)):module.sha(p.read_bytes()) for p in parent.rglob('*') if p.is_file()},
        layout=dict(configs=['config.json','run.py','RUNTIME-SOURCES.json','PREPARED.json'],services={'a':'service-a.json'}))
    write(parent/'PREPARED.json',m)
    for name in [p for _,p in MODULES]+[OBJECTIVE]:
        p=snapshot/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    (snapshot/'forward.py').write_bytes(b'pass\n')
    files={str(p.relative_to(snapshot)):module.sha(p.read_bytes()) for p in snapshot.rglob('*') if p.is_file()}
    write(snapshot/'FILES.json',files)
    report=tmp_path/'report.json'
    write(report,dict(source_manifest_sha256=module.sha((snapshot/'FILES.json').read_bytes()),
        counts=dict(tests=47,failures=0,errors=0,skipped=0),image='fixed',
        state=dict(ExitCode=0,OOMKilled=False),terminal=dict(exit_code=0)))
    bootstrap=ROOT/'tools/reports/context_objective_bootstrap.py'
    c=dict(parent=dict(output=str(parent),manifest_sha256=module.sha((parent/'PREPARED.json').read_bytes())),
        extension_source=str(snapshot),extension_cpu_report=dict(path=str(report),sha256=module.sha(report.read_bytes())),
        bootstrap=dict(path=str(bootstrap),sha256=module.sha(bootstrap.read_bytes())),options={},
        output=str(tmp_path/'candidate'),remote_root='/data/candidate')
    monkeypatch.setattr(module.parent_api,'audit',lambda *a:dict(status='cached_training_independently_audited_not_npu_admitted',provenance={}))
    monkeypatch.setattr(module,'check_paths',lambda _:None)
    import multimodal_jev.local_checkpoint as checkpoint
    monkeypatch.setattr(checkpoint,'check_paths',lambda _:None)
    return c,parent


def test_prepare_and_independent_reconstruction_preserve_runtime_and_reference(tmp_path,monkeypatch):
    c,parent=setup(tmp_path,monkeypatch)
    before={str(p):p.read_bytes() for p in parent.rglob('*') if p.is_file()}
    c['manifest_sha256']=module.prepare(c,{})['manifest_sha256']
    result=module.audit(c,{})
    assert result['reference_binding_unchanged'] and result['forward_source_unchanged']
    assert not result['npu_admitted']
    root=Path(c['output']);cfg=json.loads((root/'config.json').read_text())
    old=json.loads((parent/'config.json').read_text())
    assert cfg['dual_runtime_identity']==old['dual_runtime_identity']
    assert cfg['reference_dataset']==old['reference_dataset']
    assert (root/'RUNTIME-SOURCES.json').read_bytes()==(parent/'RUNTIME-SOURCES.json').read_bytes()
    assert 'objective-extension/bootstrap.py' in (root/'run.py').read_text()
    assert before=={str(p):p.read_bytes() for p in parent.rglob('*') if p.is_file()}


@pytest.mark.parametrize('fault',['reward','runner','reference','identity','extra','pin'])
def test_rehashed_candidate_substitution_rejected(tmp_path,monkeypatch,fault):
    c,parent=setup(tmp_path,monkeypatch);module.prepare(c,{})
    root=Path(c['output'])
    if fault=='reward':(root/'objective-extension'/OBJECTIVE).write_text('pass\n')
    elif fault=='runner':(root/'run.py').write_text('pass\n')
    elif fault=='extra':(root/'source/extra.py').write_text('pass\n')
    else:
        cfg=json.loads((root/'config.json').read_text())
        if fault=='reference':cfg['reference_dataset']['root']='/data/other'
        elif fault=='identity':cfg['dual_runtime_identity']['source_manifest_sha256']='other'
        else:cfg['pins'].clear()
        write(root/'config.json',cfg)
    manifest=json.loads((root/'PREPARED.json').read_text())
    manifest['files']={str(p.relative_to(root)):module.sha(p.read_bytes()) for p in root.rglob('*') if p.is_file() and p!=root/'PREPARED.json'}
    write(root/'PREPARED.json',manifest);c['manifest_sha256']=module.sha((root/'PREPARED.json').read_bytes())
    with pytest.raises(ValueError,match='reconstruction'):module.audit(c,{})


def test_wrong_cpu_forward_identity_rejected_before_publication(tmp_path,monkeypatch):
    c,parent=setup(tmp_path,monkeypatch)
    snapshot=Path(c['extension_source']);manifest=json.loads((snapshot/'FILES.json').read_text())
    manifest['forward.py']='wrong';write(snapshot/'FILES.json',manifest)
    report=Path(c['extension_cpu_report']['path']);value=json.loads(report.read_text())
    value['source_manifest_sha256']=module.sha((snapshot/'FILES.json').read_bytes());write(report,value)
    c['extension_cpu_report']['sha256']=module.sha(report.read_bytes())
    with pytest.raises(ValueError,match='forward dependency'):module.prepare(c,{})
    assert not Path(c['output']).exists()
