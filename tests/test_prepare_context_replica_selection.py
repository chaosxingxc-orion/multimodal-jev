from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tools.reports import prepare_context_replica_selection as module
from test_prepare_context_train_parity_job import fixture as parity_fixture,write


def setup(tmp_path,monkeypatch):
    from multimodal_jev import local_checkpoint
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda _:None)
    monkeypatch.setattr(module,'check_paths',lambda _:None)
    c=parity_fixture(tmp_path,monkeypatch);parent=Path(c['parent']['output'])
    m=module.read(parent/'PREPARED.json');training=module.read(parent/'config.json')
    m['dual_runtime_identity']=training['dual_runtime_identity']
    for name in ('policy','adapter'):
        write(parent/'initial'/name/'MANIFEST.json',{'fixture':'old'})
    write(parent/'initial/endpoint-commit.json',{'fixture':'old-endpoint'})
    snapshot=tmp_path/'snapshot';write(snapshot/'MANIFEST.json',{'policy_id':'point32'})
    point=dict(step=32,policy_id='point32',snapshot=str(snapshot))
    c.update(source_step=32,source_plan=str(tmp_path/'plan.json'),source_plan_sha256='plan',
        selection_inputs=dict(cache_root='/data/selection-cache',cache_audit={},cache_receipts={}))
    write(Path(c['source_plan']),{})
    for key in ('cache_audit','cache_receipts'):
        p=tmp_path/(key+'.json');write(p,{'fixture':key})
        c['selection_inputs'][key]=dict(path=str(p),sha256=module.file_hash(p))
    bundle={'fixture':'NPU source substituted'};plan={'fixture':'source-plan'}
    monkeypatch.setattr(module,'inputs',lambda *a:(plan,{'fixture':'parent'},parent,m,training,point,bundle,{'fixture':'panel'},{}))
    monkeypatch.setattr(module,'export_native_adapter',lambda path,b:write(path/'adapter_model.safetensors',{'fixture':'adapter'}))
    return c,parent


def test_prepare_and_independent_reconstruction_preserve_model_sources(tmp_path,monkeypatch):
    c,parent=setup(tmp_path,monkeypatch)
    before={str(p):module.file_hash(p) for p in parent.rglob('*') if p.is_file()}
    c['manifest_sha256']=module.prepare(c,{})['manifest_sha256']
    assert before=={str(p):module.file_hash(p) for p in parent.rglob('*') if p.is_file()}
    root=Path(c['output']);m=module.read(root/'PREPARED.json')
    assert m['source_step']==32 and m['recordings']==9226 and m['actor_updates']==0
    assert not (root/'initial/endpoint-commit.json').exists()
    assert module.read(root/'parity-hf.json')['policy_id']=='point32'
    assert module.read(root/'parity-macro.json')['encoded_audio_cache']=='/data/selection-cache'
    assert (root/'RUNTIME-SOURCES.json').read_bytes()==(parent/'RUNTIME-SOURCES.json').read_bytes()
    for name in m['layout']['services'].values():
        assert module.read(root/name)['policy']['policy_id']=='point32'
    proof=module.audit(c,{})
    assert proof['source_step']==32 and not proof['npu_executed']
    assert not list(root.parent.glob('selection-reconstruct-*'))


@pytest.mark.parametrize('fault',['policy','scope','runtime','client','input','manifest'])
def test_rehashed_mutation_is_rejected(tmp_path,monkeypatch,fault):
    c,parent=setup(tmp_path,monkeypatch);module.prepare(c,{})
    root=Path(c['output']);m=module.read(root/'PREPARED.json')
    if fault=='policy':write(root/'initial/policy/MANIFEST.json',{'policy_id':'other'})
    elif fault=='runtime':(root/'source/test.py').write_text('different')
    elif fault=='client':(root/'selection-client.py').write_text('pass')
    elif fault=='input':write(root/'selection-inputs/panel.json',{'gold':'invalid'})
    elif fault=='scope':
        value=module.read(root/'parity-macro.json');value['macro_queries']=32;write(root/'parity-macro.json',value)
    else:m['source_step']=318
    m['files']={str(p.relative_to(root)):module.file_hash(p) for p in root.rglob('*') if p.is_file() and p.name!='PREPARED.json'}
    write(root/'PREPARED.json',m);c['manifest_sha256']=module.file_hash(root/'PREPARED.json')
    with pytest.raises(ValueError):module.audit(c,{})


@pytest.mark.parametrize('code',[0,5])
def test_actual_runner_waits_hf_then_launches_replica_services_and_pinned_client(tmp_path,monkeypatch,code):
    from multimodal_jev import local_checkpoint,local_runtime_storage
    from tools.jev.context_process_supervisor import stop_owned
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda _:None)
    monkeypatch.setattr(local_runtime_storage,'require_training_storage',lambda _:None)
    m=dict(training_devices=[4,5],inference_services=[dict(name='a',device=6),dict(name='b',device=7)])
    write(tmp_path/'parity-hf.json',dict(expected_population=9226,source_step=32,process_device_scope={'visible_ordinal':'0'}))
    for name in ('a','b'):write(tmp_path/f'service-{name}.json',{'socket_path':str(tmp_path/name/'socket')})
    script=tmp_path/'selection-client.py'
    script.write_text('import json,os\nfrom pathlib import Path\nr=Path(__file__).parent\n'
        +"assert os.environ['ASCEND_RT_VISIBLE_DEVICES']=='0'\n(r/'hf').mkdir()\n"
        +"(r/'hf/terminal.json').write_text(json.dumps({'exit_code':"+str(code)+"}))\n"
        +"(r/'hf/reference.json').write_text('{}')\nraise SystemExit("+str(code)+')\n')
    calls=[]
    def supervise(**kw):
        record=module.read(tmp_path/'hf-process/terminal.json')
        import os
        with pytest.raises(ProcessLookupError):os.kill(record['pid'],0)
        calls.append(kw);return {'passed':True}
    monkeypatch.setitem(sys.modules,'context_process_supervisor',SimpleNamespace(supervise_services=supervise,stop_owned=stop_owned))
    (tmp_path/'cache').mkdir()
    generated=module.runner(m).replace("Path('/jev-tmp')",'Path('+repr(str(tmp_path/'cache'))+')')
    with pytest.raises(SystemExit) as e:exec(compile(generated,'<selection-runner>','exec'),{'__file__':str(tmp_path/'run.py')})
    assert e.value.code==code and len(calls)==int(code==0)
    if calls:
        assert all('context_replica_runtime.py' in s['command'][1] for s in calls[0]['services'])
        assert 'selection-client.py' in calls[0]['actor_command'][2]
        assert [s['env']['ASCEND_RT_VISIBLE_DEVICES'] for s in calls[0]['services']]==['2','3']
