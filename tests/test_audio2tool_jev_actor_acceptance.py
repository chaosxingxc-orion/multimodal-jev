import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from multimodal_jev.audio2tool_acceptance import IMAGE
from multimodal_jev.baseline import file_hash


def cli():
    directory=Path(__file__).parents[1]/'tools/jev'
    sys.path.insert(0,str(directory))
    spec=importlib.util.spec_from_file_location('jev_actor_acceptance_cli',directory/'accept_and_score_audio2tool_jev.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def fixture(tmp_path):
    source=tmp_path/'source';(source/'src/multimodal_jev').mkdir(parents=True);(source/'tools/jev').mkdir(parents=True)
    runner=source/'tools/jev/infer_audio2tool_jev.py';runner.write_text('# synthetic runner')
    module=source/'src/multimodal_jev/fixture.py';module.write_text('# synthetic module')
    run=tmp_path/'experiment/run'
    keys=['release','context-run','registry','inputs','budgets','output','root','model','asset-audit',
          'native-processor-run','capacity-root','capacity-inspection','capacity-intent-hash',
          'capacity-inspection-hash','capacity-container-id']
    cmd=['timeout','--signal=TERM','--kill-after=30s','86400s','python3','-B',str(runner)]
    for k in keys:cmd+=['--'+k,str(run) if k=='output' else 'fixture']
    binds=[f'{source}:{source}:ro',f'{run.parent}:{run.parent}:rw']
    argv=['docker','create','--device','/dev/davinci6','--device','/dev/davinci7']
    for v in binds:argv+=['-v',v]
    argv += [IMAGE,*cmd]
    plan=dict(argv=argv,image=IMAGE,automatic_retry=False,physical_devices=[6,7],wall_seconds=86400,
              source_root=str(source),source_files={str(p.relative_to(source)):file_hash(p) for p in (runner,module)})
    cid='a'*64
    item=dict(Id=cid,Image=IMAGE,Config=dict(Cmd=cmd,Env=[]),
        State=dict(Status='exited',ExitCode=0,Running=False,Restarting=False,Dead=False,Error='',OOMKilled=False),
        HostConfig=dict(NanoCpus=16*10**9,Memory=128*1024**3,ShmSize=8*1024**3,NetworkMode='host',Privileged=False,
            Devices=[dict(PathOnHost='/dev/davinci6'),dict(PathOnHost='/dev/davinci7')],Binds=binds))
    return item,plan,cid


@pytest.mark.parametrize('fault',[None,'budget','source','source_manifest','output_mount','preflight','duplicates','running'])
def test_actual_full_actor_configuration_contract(tmp_path,fault):
    item,plan,cid=fixture(tmp_path);module=cli()
    if fault=='budget':plan['wall_seconds']=10
    if fault=='source':(Path(plan['source_root'])/'src/multimodal_jev/fixture.py').write_text('changed')
    if fault=='source_manifest':plan['source_files'].pop('src/multimodal_jev/fixture.py')
    if fault=='output_mount':item['HostConfig']['Binds'].pop()
    if fault=='preflight':item['Config']['Cmd'].append('--preflight-only')
    if fault=='duplicates':item['Config']['Cmd']+=['--output','duplicate']
    if fault=='running':item['State'].update(Status='running',Running=True)
    if fault is None:assert module.configuration(item,plan,cid)['output'].endswith('/experiment/run')
    else:
        with pytest.raises(ValueError):module.configuration(item,plan,cid)


def test_targets_not_opened_until_runtime_accepted(monkeypatch):
    module=cli()
    def reject(_):raise ValueError('incomplete actor')
    monkeypatch.setattr(module,'accept',reject)
    monkeypatch.setattr(module,'score_raw',lambda _:pytest.fail('gold scoring started before runtime acceptance'))
    monkeypatch.setattr(module,'file_hash',lambda _:pytest.fail('gold hash read before runtime acceptance'))
    with pytest.raises(ValueError,match='incomplete actor'):module.score(SimpleNamespace())


def test_inspect_uses_exact_id_and_ssh_argv(monkeypatch):
    module=cli();calls=[];cid='a'*64
    def check(argv,**kwargs):
        calls.append(argv);return json.dumps([dict(Id=cid)]).encode()
    monkeypatch.setattr(module.subprocess,'check_output',check)
    assert module.inspect_container(cid)['Id']==cid
    assert calls[0][-1]=='docker inspect '+cid
    with pytest.raises(ValueError):module.inspect_container('$(wrong)')
    assert len(calls)==1
