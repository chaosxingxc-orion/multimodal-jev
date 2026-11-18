import json
from pathlib import Path

import pytest

from tools import run_operations as ops
from tools import publish_retention_inventory as publisher
from tools import retain_experiment_artifacts as retention
from test_retention_inventory import fixture as inventory_fixture, revise


TEMPLATE = 'configs/portable/operations/archive-retention-pipeline.json'


def fixture(tmp_path):
    root, ledger, data, local = inventory_fixture(tmp_path)
    config = ops.read(local)
    config['macros'].update(WORKSPACE=str(Path.cwd()), RETENTION_OUTPUT=str(tmp_path/'plan'),
        RETENTION_PIPELINE_OUTPUT=str(tmp_path/'workflow'), RETENTION_INVENTORY='/stale',
        RETENTION_INVENTORY_SHA256='stale', RETENTION_PIN_MANIFESTS=[], RETENTION_PROTECTED_PATHS=[])
    local.write_text(json.dumps(config))
    return root, ledger, data, local


def mock_storage(monkeypatch):
    for m in (ops, publisher, retention): monkeypatch.setattr(m,'check_paths',lambda *a:[])
    monkeypatch.setattr(retention,'delete_one',lambda *a:pytest.fail('must not delete'))


def test_archive_inventory_plan_binds_dependencies(tmp_path,monkeypatch):
    root,ledger,data,local=fixture(tmp_path);mock_storage(monkeypatch)
    pin=tmp_path/'pins.json'
    pin.write_text(json.dumps({str(root/'checkpoint1.pt'):ops.sha(root/'checkpoint1.pt')}))
    data['pin_manifests']={str(pin):ops.sha(pin)}
    data['protected_paths']=[str(root/'analysis.json')]
    revise(ledger,data,local)
    original=local.read_bytes()
    assert ops.run(TEMPLATE,local)['completed']==['inventory','retention']
    assert ops.run(TEMPLATE,local)['status']=='already_complete'
    assert original==local.read_bytes()
    bound=ops.read(tmp_path/'workflow/retention.macros.json')['macros']
    assert bound['RETENTION_INVENTORY_SHA256']==ops.sha(tmp_path/'publication/inventory.json')
    assert bound['RETENTION_PIN_MANIFESTS']==[str(pin)]
    assert bound['RETENTION_PROTECTED_PATHS']==[str(root/'analysis.json')]
    result=ops.read(tmp_path/'plan/COMPLETE.json')['summary']
    assert result['candidates']==0 and result['kept_checkpoint_steps']==[1,2,3]
    assert all((root/r['path']).exists() for r in data['files'])


@pytest.mark.parametrize('target',['candidate','archive','bindings','terminal'])
def test_reuse_rejects_changed_evidence(tmp_path,monkeypatch,target):
    root,_,data,local=fixture(tmp_path);mock_storage(monkeypatch)
    ops.run(TEMPLATE,local)
    archive=next(iter(ops.read(tmp_path/'publication/inventory.json')['conclusion_archives'].values()))
    path={'candidate':root/'analysis.json','archive':Path(archive),
          'bindings':tmp_path/'publication/retention-inputs.json',
          'terminal':Path(data['jobs']['transfer']['receipt'])}[target]
    path.write_text('changed')
    with pytest.raises(ValueError,match='Source changed'):ops.run(TEMPLATE,local)


def test_failed_inventory_prevents_downstream(tmp_path,monkeypatch):
    _,ledger,data,local=fixture(tmp_path);mock_storage(monkeypatch)
    data['closed']=False;revise(ledger,data,local)
    with pytest.raises(ValueError,match='Closed producer'):ops.run(TEMPLATE,local)
    assert not (tmp_path/'plan').exists()
    assert ops.read(tmp_path/'workflow/status.json')['completed']==[]
    with pytest.raises(ValueError,match='Partial workflow'):ops.run(TEMPLATE,local)


def test_direct_dependency_and_version_required():
    config=ops.read(TEMPLATE);config['schema']='local-operations-v4'
    with pytest.raises(ValueError,match='v5'):ops.ordered_steps(config)
    config=ops.read(TEMPLATE);config['steps'][1]['depends_on']=[]
    with pytest.raises(ValueError,match='direct'):ops.ordered_steps(config)


def test_mutated_publication_rejected_before_binding(tmp_path,monkeypatch):
    _,_,_,local=fixture(tmp_path);mock_storage(monkeypatch)
    ops.run(TEMPLATE,local)
    p=tmp_path/'publication/COMPLETE.json';digest=ops.sha(p);p.write_text('{}')
    with pytest.raises(ValueError,match='changed before binding'):
        ops.child_bindings({'inventory_from':'inventory'},local,{'inventory':(str(p),digest)})


@pytest.mark.parametrize('fault',['outside','extra','missing','wrong_binding'])
def test_inventory_publication_contract(tmp_path,monkeypatch,fault):
    _,_,_,local=fixture(tmp_path);mock_storage(monkeypatch)
    ops.run(TEMPLATE,local)
    p=tmp_path/'publication/COMPLETE.json';data=ops.read(p)
    if fault in ('outside','extra'):
        target=tmp_path/('outside.json' if fault=='outside' else 'publication/extra.json')
        target.write_text('{}');data['outputs'][str(target)]=ops.sha(target)
    elif fault=='missing':del data['outputs'][str(tmp_path/'publication/retention-inputs.json')]
    else:
        target=tmp_path/'publication/retention-inputs.json';value=ops.read(target)
        value['inventory_sha256']='0'*64;target.write_text(json.dumps(value))
        data['outputs'][str(target)]=ops.sha(target)
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError):ops.verified_publication('retention_inventory',p)
