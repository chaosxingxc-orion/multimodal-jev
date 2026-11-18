import fcntl
import json
from pathlib import Path

import pytest

from tools import publish_retention_inventory as publisher
from tools import retain_experiment_artifacts as retention
from test_artifact_retention import fixture as retention_fixture


TEMPLATE = 'configs/portable/operations/retention-inventory.json'


def fixture(tmp_path):
    root, _, inventory, _ = retention_fixture(tmp_path)
    jobs, audits = {}, {}
    for name in ('training', 'selection', 'transfer'):
        handle = name + '-handle'
        receipt = tmp_path/(name+'-terminal.json')
        receipt.write_text(json.dumps(dict(container_id=handle,
            state=dict(Status='exited', ExitCode=0, OOMKilled=False, Running=False))))
        audit = tmp_path/(name+'-audit.json'); audit.write_text('{"passed":true}')
        jobs[name] = dict(receipt=str(receipt), sha256=publisher.sha(receipt), container_id=handle)
        audits[name] = dict(path=str(audit), sha256=publisher.sha(audit))
    data = dict(schema='experiment-closure-ledger-v1', root=str(root), closed=True,
        dependencies_registered=True, terminal=inventory['terminal'], terminal_sha256=inventory['terminal_sha256'],
        jobs=jobs, required_jobs=list(jobs), audits=audits, required_audits=list(audits),
        files=inventory['files'], selected_checkpoint_steps=[2], pin_manifests={}, protected_paths=[])
    ledger = tmp_path/'closure.json'; ledger.write_text(json.dumps(data))
    local = tmp_path/'local.json'
    macros=dict(RETENTION_ROOT=str(root), CLOSURE_LEDGER=str(ledger),
        CLOSURE_LEDGER_SHA256=publisher.sha(ledger), INVENTORY_PUBLICATION_OUTPUT=str(tmp_path/'publication'),
        RETENTION_LOCK=str(tmp_path/'experiment.lock'))
    local.write_text(json.dumps(dict(schema='jev-local-macros-v1', macros=macros)))
    return root, ledger, data, local


def revise(ledger, data, local):
    ledger.write_text(json.dumps(data))
    config=publisher.read(local)
    config['macros']['CLOSURE_LEDGER_SHA256']=publisher.sha(ledger)
    local.write_text(json.dumps(config))


def test_archive_inventory_and_retention_interoperate(tmp_path, monkeypatch):
    root, ledger, data, local = fixture(tmp_path)
    monkeypatch.setattr(publisher, 'check_paths', lambda *a: [])
    result = publisher.run(TEMPLATE, local)
    assert result['archived_conclusions']==1 and result['candidates']==2
    assert publisher.run(TEMPLATE, local)['status']=='already_complete'
    inventory=publisher.read(result['inventory'])
    archived=Path(inventory['conclusion_archives']['conclusion.md'])
    assert archived.read_bytes()==(root/'conclusion.md').read_bytes()
    inputs=publisher.read(tmp_path/'publication/retention-inputs.json')
    config=dict(inputs,schema='closed-experiment-retention-v1',output_directory=str(tmp_path/'plan'),
                max_files=100,max_bytes=10000)
    plan,_=retention.prepare(config)
    assert plan['kept_checkpoint_steps']==[2,3]
    assert {r['relative'] for r in plan['files']}=={'analysis.json','checkpoint1.pt'}
    assert all((root/row['path']).exists() for row in data['files'])


@pytest.mark.parametrize('fault', ['live','oom','failed','missing_job','missing_audit',
    'failed_audit','unregistered','open','changed_file','escape','no_conclusion','bad_terminal'])
def test_refuse_incomplete_or_mismatched_closure(tmp_path,monkeypatch,fault):
    root,ledger,data,local=fixture(tmp_path)
    monkeypatch.setattr(publisher,'check_paths',lambda *a:[])
    if fault in ('live','oom','failed'):
        row=data['jobs']['transfer']; path=Path(row['receipt']);value=publisher.read(path)
        if fault=='live':value['state']['Status']='running'
        elif fault=='oom':value['state']['OOMKilled']=True
        else:value['state']['ExitCode']=1
        path.write_text(json.dumps(value));row['sha256']=publisher.sha(path)
    elif fault=='missing_job':del data['jobs']['transfer']
    elif fault=='missing_audit':del data['audits']['selection']
    elif fault=='failed_audit':
        row=data['audits']['selection'];path=Path(row['path']);path.write_text('{"passed":false}')
        row['sha256']=publisher.sha(path)
    elif fault=='unregistered':data['dependencies_registered']=False
    elif fault=='open':data['closed']=False
    elif fault=='changed_file':(root/'analysis.json').write_text('changed')
    elif fault=='escape':data['files'][0]['path']='../outside'
    elif fault=='no_conclusion':data['files']=[r for r in data['files'] if r['role']!='conclusion']
    elif fault=='bad_terminal':
        (root/'terminal.json').write_text('{"exit_code":1}')
        data['terminal_sha256']=publisher.sha(root/'terminal.json')
    revise(ledger,data,local)
    with pytest.raises(ValueError):publisher.run(TEMPLATE,local)
    assert not (tmp_path/'publication/COMPLETE.json').exists()
    assert (root/'checkpoint1.pt').exists()
    with pytest.raises(ValueError,match='Partial'):publisher.run(TEMPLATE,local)


def test_registered_pin_preserves_complete_step(tmp_path,monkeypatch):
    root,ledger,data,local=fixture(tmp_path)
    pin=tmp_path/'pins.json';pin.write_text(json.dumps({str(root/'checkpoint1.pt'):publisher.sha(root/'checkpoint1.pt')}))
    data['pin_manifests']={str(pin):publisher.sha(pin)};revise(ledger,data,local)
    monkeypatch.setattr(publisher,'check_paths',lambda *a:[])
    result=publisher.run(TEMPLATE,local)
    assert result['kept_checkpoint_steps']==[1,2,3] and result['candidates']==1
    pin.write_text('{}')
    with pytest.raises(ValueError,match='identity changed'):publisher.run(TEMPLATE,local)


def test_shared_experiment_lock_and_archive_tamper(tmp_path,monkeypatch):
    _,_,_,local=fixture(tmp_path)
    monkeypatch.setattr(publisher,'check_paths',lambda *a:[])
    with (tmp_path/'experiment.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):publisher.run(TEMPLATE,local)
    result=publisher.run(TEMPLATE,local)
    archive=next(iter(publisher.read(result['inventory'])['conclusion_archives'].values()))
    Path(archive).write_text('tampered')
    with pytest.raises(ValueError,match='identity changed'):publisher.run(TEMPLATE,local)
