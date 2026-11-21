import json
from pathlib import Path
import pytest
from tools.reports import minimal_training_result as m


def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'check_paths',lambda _:None)
    root=tmp_path/'collection';root.mkdir();out=tmp_path/'capsule';assets=tmp_path/'assets';assets.mkdir()
    def put(p,x):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x));return m.sha(p)
    files={}
    files['config.json']=put(root/'config.json',{'model':'shared'})
    files['run/result.json']=put(root/'run/result.json',{'passed':True})
    for rank in range(2):
        name=f'run/rank-{rank}/update-0001.json';files[name]=put(root/name,{'update':1,'metrics':{'loss':[.2,.3]},'rollout':{'queries':[{'secret':'large trajectory'}],'total_seconds':3},'real_queries':2})
    files['run/execution-0001.pt']=put(root/'run/execution-0001.pt',{'temporary':'weight'})
    ah={}
    for name in ['run/policy-0001/execution.pt','run/adapter-0001/adapter.json']:
        ah[name]=put(assets/name,{'final':'weight'});files[name]=put(root/name,{'final':'weight'})
    report=tmp_path/'assets-report.json';rh=put(report,{'root':str(assets),'files':ah})
    archive=tmp_path/'collected.tar';archive.write_bytes(b'archive')
    transport=tmp_path/'transport.json';put(transport,{'status':'transport_complete_independent_audit_required','state':{'Running':False,'ExitCode':0},'files':files,'container_id':'test','archive_sha256':m.sha(archive)})
    terminal=tmp_path/'terminal.json';put(terminal,{'exit_code':0,'status':'audited'})
    return dict(schema='minimal-training-result-v1',interval_seconds=900,collection=str(root),output=str(out),transport=str(transport),archive=str(archive),world_size=2,final_step=1,io_workers=2,max_delete_bytes=100000,max_delete_files=100,completion={'path':str(terminal),'equals':{'exit_code':0,'status':'audited'}},owner={'pid':99999999,'start_ticks':'0'},retain_files={},selection_assets={'path':str(report),'sha256':rh},references={'base':'shared'},protected_paths=[str(assets)],audit_reports=[str(terminal)])


def test_capsule_rebuilds_curve_after_raw_deleted(tmp_path,monkeypatch):
    c=fixture(tmp_path,monkeypatch);m.prepare(c);r=m.apply(c)
    assert r['deleted_files']==6 and r['deleted_bytes']>0
    assert not Path(c['archive']).exists()
    out=Path(c['output']);curve=m.read(out/'training-curve.json')
    assert len(curve['rows'])==2 and curve['rows'][0]['metrics']=={'loss':[.2,.3]}
    assert 'queries' not in curve['rows'][0]['rollout']
    assert m.apply(c)==r
    assert (out/'model/policy-0001/execution.pt').exists()

@pytest.mark.parametrize('fault',['pending','failed','changed_curve','changed_archive','changed_asset','protected','budget'])
def test_rejects_unsafe_deletion(tmp_path,monkeypatch,fault):
    c=fixture(tmp_path,monkeypatch);root=Path(c['collection']);archive=Path(c['archive'])
    if fault=='pending':Path(c['completion']['path']).unlink();assert m.apply(c)['deleted_bytes']==0;return
    if fault=='failed':Path(c['completion']['path']).write_text('{"exit_code":1}')
    if fault=='changed_curve':(root/'run/rank-0/update-0001.json').write_text('{}')
    if fault=='changed_archive':archive.write_bytes(b'changed')
    if fault=='changed_asset':(tmp_path/'assets/run/policy-0001/execution.pt').write_text('{}')
    if fault=='protected':c['protected_paths'].append(str(root/'run'))
    if fault=='budget':c['max_delete_bytes']=1
    with pytest.raises((ValueError,KeyError)):m.apply(c)
    assert archive.exists() and (root/'run/execution-0001.pt').exists()


def test_closed_gate_still_waits_for_owner(monkeypatch,tmp_path):
    import os
    c=fixture(tmp_path,monkeypatch);c['owner']={'pid':os.getpid(),'start_ticks':Path(f'/proc/{os.getpid()}/stat').read_text().split(') ',1)[1].split()[19]}
    assert m.apply(c)['deleted_bytes']==0


def test_whitelist():
    assert m.deletable('run/groups-0001.pt')
    assert not any(m.deletable(x) for x in ['run/result.json','run/final.pt','models/model.pt','run/rank-0/config.json'])
