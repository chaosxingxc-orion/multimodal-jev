import copy
import json
from pathlib import Path
import sys
import shlex
import subprocess

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
from tools.jev import prepare_fsc_context_package as package
from tools.jev import launch_fsc_context_pair as launcher
from tools.jev import collect_fsc_context_pair as collector
from test_fsc_context_source import source
from tools.jev import prepare_fsc_context_source as source_module
from test_fsc_context_pair import inputs, panel, full


@pytest.fixture
def prepared(source, tmp_path, monkeypatch):
    args, _, _ = source
    source_module.prepare(**args)
    source_root=args['output'];source_manifest=source_module.read(source_root/'PACKAGE.json')
    model=tmp_path/'model';model.mkdir();(model/'model.safetensors').write_bytes(b'base fixture')
    (model/'config.json').write_text('{}')
    monkeypatch.setattr(package,'MODEL',str(model))
    manifest=tmp_path/'inputs/MANIFEST.json';manifest.parent.mkdir();manifest.write_text('{}')
    requests=manifest.parent/'requests.jsonl';requests.write_text('{}\n')
    receipts=manifest.parent/'receipts.jsonl';receipts.write_text('{}\n')
    ins=dict(input_manifest=dict(path=str(manifest),sha256=package.sha(manifest)),
             input_audit=dict(path=str(args['input_audit']),sha256=args['input_audit_sha256']),
             requests=str(requests),input_receipts=str(receipts),cache_root=package.CACHE)
    monkeypatch.setattr(package,'load_inputs',lambda c:([],[],dict(manifest={'processor_sha256':source_manifest['processor_sha256']})))
    spec=dict(source_package=str(source_root),source_manifest_sha256=package.sha(source_root/'PACKAGE.json'),inputs=ins)
    monkeypatch.chdir(tmp_path)
    Path('src').mkdir();Path('src/model.py').write_text('fixture')
    Path('tools/jev').mkdir(parents=True);Path('tools/jev/run_fsc_context_pair.py').write_text('fixture')
    out=tmp_path/'runtime';remote=package.PREFIX+'fsc-context-pair-test'
    package.prepare(spec,out,remote)
    return spec,out,package.sha(out/'PACKAGE.json')


def test_real_policy_copy_readonly_launch_and_short_ipc(prepared):
    spec,out,digest=prepared;m=package.verify(out,digest);c=package.read(out/'config.json')
    assert (out/'frozen-source/initial/policy/execution.pt').read_bytes() == (Path(spec['source_package'])/'initial/policy/execution.pt').read_bytes()
    argv=launcher.command(m,c)
    def values(flag):return [argv[i+1] for i,v in enumerate(argv) if v==flag]
    assert values('--entrypoint')==['/opt/multimodal-jev/runtime-entrypoint.sh']
    assert package.CACHE+':'+package.CACHE+':ro' in values('-v')
    assert 'VLLM_RPC_BASE_PATH=/jev-tmp' in values('-e')
    assert 'TMPDIR=/jev-tmp' in values('-e')
    assert values('--device')==['/dev/'+d+':/dev/'+d for d in ('davinci7','davinci_manager','devmm_svm','hisi_hdc')]
    assert argv[-2:]==['--output',m['remote_root']+'/evaluation']


@pytest.mark.parametrize('fault',['source','input','code','config','extra','symlink','base'])
def test_runtime_package_corruption_rejected(prepared,fault):
    _,out,digest=prepared
    names=dict(source='frozen-source/selected/policy/execution.pt',input='inputs/requests.jsonl',
               code='source/src/model.py',config='config.json',extra='unexpected')
    if fault=='symlink':(out/'link').symlink_to(out/'config.json')
    elif fault=='base':Path(package.MODEL,'model.safetensors').write_bytes(b'changed')
    else:(out/names[fault]).write_bytes(b'changed')
    with pytest.raises(ValueError):package.verify(out,digest)


def test_plan_does_not_call_remote(prepared,monkeypatch,capsys):
    _,out,digest=prepared
    monkeypatch.setattr(sys,'argv',['launch','--package',str(out),'--manifest-sha256',digest])
    def forbidden(*a,**k):raise AssertionError('remote call forbidden')
    monkeypatch.setattr(launcher.subprocess,'check_output',forbidden)
    monkeypatch.setattr(launcher.subprocess,'run',forbidden)
    launcher.main()
    assert json.loads(capsys.readouterr().out)['status']=='validated_only_no_remote_calls'


def test_uncertain_start_wait_preserves_exact_handle_and_lease(prepared,monkeypatch,tmp_path):
    import fcntl
    _,out,digest=prepared
    local=Path('.local/decision-objective-research');local.mkdir(parents=True)
    lease=Path('.local/simple-final-v1/npu175-evaluation-training.lock');lease.parent.mkdir(parents=True)
    monkeypatch.setattr(sys,'argv',['launch','--package',str(out),'--manifest-sha256',digest,'--launch'])
    created=[];waits=[];sleeps=[]
    def held():
        with lease.open('a') as f:
            with pytest.raises(BlockingIOError):fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def output(argv,**kw):
        held();cmd=shlex.split(argv[-1])
        if cmd[:3]==['docker','image','inspect']:return json.dumps([{'Id':launcher.IMAGE}])
        if cmd[:2]==['npu-smi','info']:return 'No running processes found in NPU 7'
        if cmd[:2]==['python3','-c']:return 'False' if '.exists()' in cmd[-1] else ''
        if cmd[:2]==['docker','create']:created.append(cmd);return 'c'*64
        return ''
    def run(argv,**kw):
        held()
        if argv[0]=='scp':return subprocess.CompletedProcess(argv,0,'','')
        cmd=shlex.split(argv[-1])
        if cmd[:2]==['docker','start']:
            assert package.read(local/'fsc-context-pair-test-launch.json')['container_id']=='c'*64
            return subprocess.CompletedProcess(argv,255,'','unknown observation')
        assert cmd==['docker','wait','c'*64];waits.append(cmd)
        return subprocess.CompletedProcess(argv,255 if len(waits)==1 else 0,'' if len(waits)==1 else '1\n','')
    monkeypatch.setattr(launcher.subprocess,'check_output',output)
    monkeypatch.setattr(launcher.subprocess,'run',run)
    monkeypatch.setattr(launcher.time,'sleep',lambda seconds:(held(),sleeps.append(seconds)))
    launcher.main()
    assert len(created)==1 and len(waits)==2 and sleeps==[60]*10
    with pytest.raises(ValueError,match='existing launch'):launcher.main()


@pytest.fixture
def collected(prepared,inputs,tmp_path,monkeypatch):
    _,out,digest=prepared;m=package.verify(out,digest);spec=package.read(out/'config.json')
    rows,receipts=inputs
    monkeypatch.setattr(collector,'load_inputs',lambda c:(rows,receipts,{}))
    argv=launcher.command(m,spec)
    def values(flag):return [argv[i+1] for i,v in enumerate(argv) if v==flag]
    launch=dict(container_id='container',image=m['image'],remote_root=m['remote_root'],argv=argv,
                config_sha256=package.sha(out/'config.json'),manifest_sha256=digest)
    state=dict(Status='exited',ExitCode=0,OOMKilled=False)
    observed=dict(Id='container',Image=m['image'],State=state,
        Config=dict(Entrypoint=values('--entrypoint'),Cmd=argv[argv.index(m['image'])+1:],Env=values('-e')),
        HostConfig=dict(Binds=values('-v'),NetworkMode='none',Devices=[
            dict(PathOnHost=v.split(':')[0],PathInContainer=v.split(':')[1],CgroupPermissions='rwm')
            for v in values('--device')]))
    root=tmp_path/'collected';root.mkdir();(root/'config.json').write_bytes((out/'config.json').read_bytes())
    ev=root/'evaluation';ev.mkdir();(ev/'input.json').write_bytes((out/'config.json').read_bytes())
    source_manifest=package.read(out/'frozen-source/PACKAGE.json');panels={};results={}
    for role,entry in source_manifest['policies'].items():
        c=dict(spec['inputs'],model=spec['model'],base_id=entry['contract']['base_id'],
               snapshot=spec['source_package']+'/'+entry['policy'],adapter=spec['source_package']+'/'+entry['adapter'],
               policy_id=entry['policy_id'],contract=entry['contract'],source_role=role,
               source_selection=dict(path=spec['source_package']+'/source-selection.json',sha256=package.sha(out/'frozen-source/source-selection.json')),
               physical_devices=spec['physical_devices'],pins=spec['pins'])
        folder=ev/role;folder.mkdir();(folder/'config.json').write_text(json.dumps(c))
        panel(folder/'parity',c,rows,receipts)
        # Match the real package input identity, rather than the generic fixture string.
        for phase in ('hf','vllm'):
            p=folder/'parity'/(phase+'.json');v=package.read(p)
            v['input_manifest_sha256']=c['input_manifest']['sha256'];p.write_text(json.dumps(v))
        full(folder/'inference',c,rows,receipts)
        panels[role]=package.read(folder/'parity/comparison.json');results[role]=package.read(folder/'inference/result.json')
    (ev/'terminal.json').write_text(json.dumps(dict(exit_code=0)))
    (ev/'result.json').write_text(json.dumps(dict(status='pair_inference_complete_independent_collection_pending',
        source_manifest_sha256=spec['source_manifest_sha256'],panels=panels,policies=results,target_tuning=False)))
    transport=dict(status='transport_complete_independent_audit_required',container_id='container',state=copy.deepcopy(state),
        image=m['image'],remote_root=m['remote_root'],files={str(p.relative_to(root)):package.sha(p) for p in root.rglob('*') if p.is_file()})
    return dict(package=out,digest=digest,launch=launch,observed=observed,collection=root,transport=transport)


def test_independent_collector_recomputes_complete_pair(collected):
    result=collector.audit(**collected)
    assert result['status']=='fsc_pair_inference_independently_collected'
    assert set(result['predictions'])=={'initial','selected'}
    assert result['policies']['selected']['recordings']==3793


@pytest.mark.parametrize('fault',['image','device','env','mount','entrypoint','exit','transport','artifact','summary'])
def test_actual_runtime_and_outputs_cannot_be_substituted(collected,fault):
    a=collected;o=a['observed']
    if fault=='image':o['Image']='other'
    elif fault=='device':o['HostConfig']['Devices'].pop()
    elif fault=='env':o['Config']['Env'].append('TMPDIR=/long/changed')
    elif fault=='mount':o['HostConfig']['Binds'].pop()
    elif fault=='entrypoint':o['Config']['Entrypoint']=['python']
    elif fault=='exit':o['State']['ExitCode']=1
    elif fault=='transport':a['transport']['state']['Status']='running'
    else:
        path=a['collection']/('evaluation/result.json' if fault=='summary' else 'evaluation/selected/inference/predictions.json')
        value=package.read(path)
        if fault=='summary':value['target_tuning']=True
        else:value['records'].pop()
        path.write_text(json.dumps(value));a['transport']['files'][str(path.relative_to(a['collection']))]=package.sha(path)
    with pytest.raises(ValueError):collector.audit(**a)
