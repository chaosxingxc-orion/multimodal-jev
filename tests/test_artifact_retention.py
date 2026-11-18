import json
from pathlib import Path
import pytest
from tools import retain_experiment_artifacts as module


def fixture(tmp_path):
    root=tmp_path/'experiment';root.mkdir()
    entries=[]
    for name,role,step in [('checkpoint1.pt','checkpoint',1),('checkpoint2.pt','checkpoint',2),
        ('checkpoint3.pt','checkpoint',3),('analysis.json','intermediate',None),
        ('config.json','config',None),('result.json','final',None),('conclusion.md','conclusion',None)]:
        p=root/name;p.write_text(name)
        entries.append(dict(path=name,sha256=module.sha(p),role=role,step=step))
    (root/'unlisted.tmp').write_text('keep unlisted')
    (root/'terminal.json').write_text('{"exit_code":0}')
    archive=tmp_path/'archived-conclusion.md';archive.write_bytes((root/'conclusion.md').read_bytes())
    inventory=tmp_path/'inventory.json'
    data=dict(schema='closed-experiment-inventory-v1',closed=True,evaluation_complete=True,
        terminal='terminal.json',terminal_sha256=module.sha(root/'terminal.json'),files=entries,
        conclusion_archives={'conclusion.md':str(archive)},selected_checkpoint_steps=[2])
    inventory.write_text(json.dumps(data))
    config=dict(schema='closed-experiment-retention-v1',root=str(root),inventory=str(inventory),
        inventory_sha256=module.sha(inventory),pin_manifests=[],protected_paths=[],
        output_directory=str(tmp_path/'publication'),lock_path=str(tmp_path/'lock'),max_files=10,max_bytes=10000)
    return root,inventory,data,config


def test_preserves_latest_selected_results_and_unlisted(tmp_path):
    root,_,_,config=fixture(tmp_path)
    plan,_=module.prepare(config)
    assert {Path(r['path']).name for r in plan['files']}=={'checkpoint1.pt','analysis.json'}
    assert plan['kept_checkpoint_steps']==[2,3]


def test_dependency_protects_entire_checkpoint_step(tmp_path):
    root,inventory,data,config=fixture(tmp_path)
    p=root/'checkpoint1-other-rank.pt';p.write_text('rank1')
    data['files'].append(dict(path=p.name,sha256=module.sha(p),role='checkpoint',step=1))
    inventory.write_text(json.dumps(data));config['inventory_sha256']=module.sha(inventory)
    pins=root.parent/'pins.json';pins.write_text(json.dumps({str(root/'checkpoint1.pt'):module.sha(root/'checkpoint1.pt')}))
    config['pin_manifests']=[str(pins)]
    plan,_=module.prepare(config)
    assert [r['relative'] for r in plan['files']]==['analysis.json']
    assert plan['kept_checkpoint_steps']==[1,2,3]


@pytest.mark.parametrize('fault',['live','evaluation','archive','changed','budget','escape','symlink'])
def test_rejects_before_deletion(tmp_path,fault):
    root,inventory,data,config=fixture(tmp_path)
    if fault=='live':data['closed']=False
    elif fault=='evaluation':data['evaluation_complete']=False
    elif fault=='archive':Path(data['conclusion_archives']['conclusion.md']).write_text('wrong')
    elif fault=='changed':(root/'analysis.json').write_text('wrong')
    elif fault=='budget':config['max_files']=1
    elif fault=='escape':data['files'][0]['path']='../outside'
    elif fault=='symlink':
        (root/'analysis.json').unlink();(root/'analysis.json').symlink_to(root/'config.json')
    inventory.write_text(json.dumps(data));config['inventory_sha256']=module.sha(inventory)
    with pytest.raises(ValueError):module.prepare(config)
    assert (root/'checkpoint1.pt').exists()


def test_apply_journal_and_reuse(tmp_path,monkeypatch):
    root,_,_,config=fixture(tmp_path)
    t=tmp_path/'template.json';t.write_text(json.dumps(config))
    local=tmp_path/'local.json';local.write_text(json.dumps(dict(schema='jev-local-macros-v1',macros={})))
    monkeypatch.setattr(module,'check_paths',lambda *a:[])
    assert module.run(t,local,True)['deleted_files']==2
    assert not (root/'checkpoint1.pt').exists()
    assert (root/'checkpoint2.pt').exists() and (root/'checkpoint3.pt').exists()
    assert (root/'unlisted.tmp').exists() and (root/'config.json').exists()
    assert len(list((tmp_path/'publication/events').glob('*-deleted.json')))==2
    assert module.run(t,local,True)['status']=='already_complete'
    (root/'analysis.json').write_text('reappeared')
    with pytest.raises(ValueError,match='reappeared'):module.run(t,local,True)


def test_failed_partial_deletion_never_retried(tmp_path,monkeypatch):
    root,_,_,config=fixture(tmp_path)
    t=tmp_path/'template.json';t.write_text(json.dumps(config))
    local=tmp_path/'local.json';local.write_text(json.dumps(dict(schema='jev-local-macros-v1',macros={})))
    monkeypatch.setattr(module,'check_paths',lambda *a:[])
    real=module.delete_one;calls=[]
    def fail_second(row,protection):
        calls.append(row)
        if len(calls)==2:raise RuntimeError('injected failure')
        real(row,protection)
    monkeypatch.setattr(module,'delete_one',fail_second)
    with pytest.raises(RuntimeError):module.run(t,local,True)
    with pytest.raises(ValueError,match='Partial'):module.run(t,local,True)
    assert len(calls)==2 and (root/'analysis.json').exists()
