"""Runtime provenance refusal checks, with model and population substitutes."""
import copy
import json
from pathlib import Path
import pytest
from tools.reports import collect_slurp_context_test_pair as collector
from tools.reports import prepare_slurp_context_test_package as package
from tools.reports import launch_slurp_context_test_pair as launcher
from test_slurp_context_test_package import prepared


@pytest.fixture
def collected(prepared,tmp_path,monkeypatch):
    out,digest,_=prepared;m=package.verify(out,digest);spec=package.read(out/'config.json')
    argv=launcher.command(m,spec)
    def values(flag):return [argv[i+1] for i,v in enumerate(argv) if v==flag]
    state=dict(Status='exited',ExitCode=0,OOMKilled=False)
    launch=dict(container_id='fixture',image=m['image'],remote_root=m['remote_root'],argv=argv,
        config_sha256=package.sha(out/'config.json'),manifest_sha256=digest)
    observed=dict(Id='fixture',Image=m['image'],State=copy.deepcopy(state),
        Config=dict(Entrypoint=values('--entrypoint'),Cmd=argv[argv.index(m['image'])+1:],Env=values('-e')),
        HostConfig=dict(Binds=values('-v'),NetworkMode='none',Devices=[
            dict(PathOnHost=v.split(':')[0],PathInContainer=v.split(':')[1],CgroupPermissions='rwm')
            for v in values('--device')]))
    root=tmp_path/'collected';root.mkdir();(root/'config.json').write_bytes((out/'config.json').read_bytes())
    ev=root/'evaluation';ev.mkdir();(ev/'input.json').write_bytes((out/'config.json').read_bytes())
    rows=[dict(id='example')];receipts=[dict(id='example',fixture=True)]
    local_pair={r:dict(policy_id=r,snapshot=str(out/'frozen-source'/r/'policy')) for r in ('initial','selected')}
    monkeypatch.setattr(collector,'configs',lambda s:(local_pair,rows,receipts))
    monkeypatch.setattr(collector,'require_parity',lambda *a:dict(exact=True))
    monkeypatch.setattr(collector,'check_full',lambda folder,c,*a:dict(policy_id=c['policy_id'],recordings=1))
    panels={};results={}
    def put(p,value):p.write_text(json.dumps(value))
    for role,c in local_pair.items():
        folder=ev/role;folder.mkdir();expected=package.relocate(c,str(out),m['remote_root'])
        put(folder/'config.json',expected);put(folder/'parity.json',dict(exact=True))
        pred=dict(id='example',policy_id=role,prob=[.1]*60)
        put(folder/'predictions.json',dict(records=[pred]))
        put(folder/'record-00000.json',dict(prediction=pred,input_receipt=receipts[0]))
        panels[role]=dict(exact=True);results[role]=dict(policy_id=role,recordings=1)
    put(ev/'terminal.json',dict(exit_code=0))
    put(ev/'result.json',dict(status='test_pair_inference_complete_independent_collection_pending',
        source_manifest_sha256=spec['source_manifest_sha256'],panels=panels,policies=results,target_tuning=False))
    transport=dict(status='transport_complete_independent_audit_required',container_id='fixture',state=state,
        image=m['image'],remote_root=m['remote_root'],files={str(p.relative_to(root)):package.sha(p) for p in root.rglob('*') if p.is_file()})
    return dict(package=out,digest=digest,launch=launch,observed=observed,collection=root,transport=transport)


def test_source_and_actual_runtime_bound_before_scoring(collected):
    r=collector.audit(**collected)
    assert r['status']=='slurp_test_pair_inference_independently_collected'
    assert set(r['predictions'])=={'initial','selected'}
    assert r['predictions']['initial']['path'].endswith('/evaluation/initial/predictions.json')


@pytest.mark.parametrize('fault',['image','device','env','mount','exit','state','config','durable','summary'])
def test_actual_provenance_and_records_cannot_be_replaced(collected,fault):
    a=collected;o=a['observed']
    if fault=='image':o['Image']='other'
    elif fault=='device':o['HostConfig']['Devices'].pop()
    elif fault=='env':o['Config']['Env'].append('TMPDIR=/other')
    elif fault=='mount':o['HostConfig']['Binds'].pop()
    elif fault=='exit':o['State']['ExitCode']=1
    elif fault=='state':a['transport']['state']['Status']='running'
    else:
        name={'config':'evaluation/selected/config.json','durable':'evaluation/selected/record-00000.json','summary':'evaluation/result.json'}[fault]
        p=a['collection']/name;v=json.loads(p.read_text())
        if fault=='config':v['policy_id']='different'
        elif fault=='durable':v['prediction']['prob'][0]=.9
        else:v['target_tuning']=True
        p.write_text(json.dumps(v));a['transport']['files'][name]=package.sha(p)
    with pytest.raises(ValueError):collector.audit(**a)
