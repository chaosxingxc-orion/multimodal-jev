import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tools.reports import prepare_context_train_parity_job as module


def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))


def settings():
    from test_prepare_context_reference_job import training
    c=training();c.update(model='/shared/model',initial_adapter='/data/bound/initial/adapter')
    m=dict(training_devices=[4,5],inference_services=[dict(name='a',device=6),dict(name='b',device=7)])
    return c,m


@pytest.mark.parametrize('code',[0,7])
def test_runner_waits_for_real_hf_child_before_starting_services(tmp_path,monkeypatch,code):
    import multimodal_jev.local_runtime_storage as storage
    import multimodal_jev.local_checkpoint as checkpoint
    from tools.jev.context_process_supervisor import stop_owned
    monkeypatch.setattr(storage,'require_training_storage',lambda _:None)
    monkeypatch.setattr(checkpoint,'check_paths',lambda _:None)
    _,m=settings();calls=[]
    def services(**kw):
        terminal=json.loads((tmp_path/'hf-process/terminal.json').read_text())
        assert terminal['exit_code']==0
        import os
        with pytest.raises(ProcessLookupError):os.kill(terminal['pid'],0)
        calls.append(kw);return {'passed':True}
    monkeypatch.setitem(sys.modules,'context_process_supervisor',SimpleNamespace(
        supervise_services=services,stop_owned=stop_owned))
    child=tmp_path/'source/tools/reports/context_train_macro_parity.py';child.parent.mkdir(parents=True)
    child.write_text('import json,os\nfrom pathlib import Path\nr=Path(__file__).parents[3]\n'
        +"assert os.environ['ASCEND_RT_VISIBLE_DEVICES']=='0'\n"
        +"(r/'hf').mkdir()\n(r/'hf/terminal.json').write_text(json.dumps({'exit_code':"+str(code)+"}))\n"
        +"(r/'hf/reference.json').write_text('{}')\nraise SystemExit("+str(code)+')\n')
    write(tmp_path/'parity-hf.json',dict(expected_population=40513,stage1_endpoint=318,
        process_device_scope={'visible_ordinal':'0'}))
    for name in ('a','b'):write(tmp_path/f'service-{name}.json',{'socket_path':str(tmp_path/name/'socket')})
    script=module.runner(m).replace("Path('/jev-tmp')",'Path('+repr(str(tmp_path/'cache'))+')')
    (tmp_path/'cache').mkdir()
    with pytest.raises(SystemExit) as exc:exec(compile(script,'<runner>','exec'),{'__file__':str(tmp_path/'run.py')})
    assert exc.value.code==code
    assert len(calls)==int(code==0)
    if calls:
        call=calls[0]
        assert call['actor_env']['ASCEND_RT_VISIBLE_DEVICES']==''
        assert [s['env']['ASCEND_RT_VISIBLE_DEVICES'] for s in call['services']]==['2','3']
        assert call['actor_command'][1]=='-c' and 'torchrun' not in str(call)


def fixture(tmp_path,monkeypatch):
    from test_prepare_context_reference_job import setup
    c=setup(tmp_path,monkeypatch);parent=Path(c['parent']['output'])
    training,m=settings();write(parent/'config.json',training)
    original=module.read(parent/'PREPARED.json');original.update(m);write(parent/'PREPARED.json',original)
    for name in ('context_train_macro_parity.py','context_train_parity_inputs.py'):
        target=parent/'source/tools/reports'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(Path(module.__file__).with_name(name).read_bytes())
    write(parent/'RUNTIME-SOURCES.json',{str(p.relative_to(parent/'source')):module.file_hash(p)
        for p in (parent/'source').rglob('*.py')})
    inputs=tmp_path/'inputs'
    write(inputs/'inputs.json',{'source':{'release_sha256':training['release_sha256'],
        'processor_sha256':training['dual_runtime_identity']['processor_sha256']}})
    write(inputs/'MANIFEST.json',{'fixture':True})
    c['inputs']=dict(output=str(inputs),manifest_sha256=module.file_hash(inputs/'MANIFEST.json'),
        cache_root=training['encoded_audio_cache'])
    monkeypatch.setattr(module,'audit_parent',lambda *a:{'fixture':'NPU-parent'})
    monkeypatch.setattr(module,'audit_inputs',lambda *a:{'fixture':'input-audit'})
    monkeypatch.setattr(module,'check_paths',lambda _:None)
    return c


def test_package_preserves_runtime_and_uses_separate_phase_outputs(tmp_path,monkeypatch):
    c=fixture(tmp_path,monkeypatch);parent=Path(c['parent']['output'])
    before={str(p):module.file_hash(p) for p in parent.rglob('*') if p.is_file()}
    result=module.prepare(c,{})
    assert before=={str(p):module.file_hash(p) for p in parent.rglob('*') if p.is_file()}
    root=Path(c['output']);hf=module.read(root/'parity-hf.json');macro=module.read(root/'parity-macro.json')
    assert not result['training_dispatch_allowed'] and not result['npu_executed']
    assert hf['physical_devices']==[4] and hf['process_device_scope']['visible_ordinal']=='0'
    assert hf['output'].endswith('/hf') and macro['output'].endswith('/evaluation')
    assert hf['macro_client']['attempt_id']=='parity-attempt'
    assert hf['macro_client']['max_queries']==128 and hf['macro_client']['rollout_queries']==32
    assert hf['train_inputs']['output']==c['remote_root']+'/parity-inputs'
    assert (root/'RUNTIME-SOURCES.json').read_bytes()==(parent/'RUNTIME-SOURCES.json').read_bytes()
    for name,digest in hf['pins'].items():
        if name.startswith(c['remote_root']+'/'):assert module.file_hash(root/Path(name).relative_to(c['remote_root']))==digest
    proof=module.audit(dict(c,manifest_sha256=result['manifest_sha256']),{'fresh_audit':True})
    assert proof['recordings']==256 and proof['actor_updates']==0
    assert not proof['npu_executed'] and not proof['numerical_admission']


@pytest.mark.parametrize('fault',['cache','release','processor','runtime-code','budget','overlap'])
def test_incompatible_bindings_fail_before_publication(tmp_path,monkeypatch,fault):
    c=fixture(tmp_path,monkeypatch);parent=Path(c['parent']['output'])
    if fault=='cache':c['inputs']['cache_root']='/data/wrong'
    elif fault in ('release','processor'):
        p=Path(c['inputs']['output'])/'inputs.json';v=module.read(p)
        v['source']['release_sha256' if fault=='release' else 'processor_sha256']='different';write(p,v)
    elif fault=='runtime-code':
        p=parent/'RUNTIME-SOURCES.json';v=module.read(p);v['tools/reports/context_train_macro_parity.py']='different';write(p,v)
    elif fault=='budget':
        p=parent/'PREPARED.json';v=module.read(p);v['training_devices']=list(range(6));write(p,v)
    else:c['remote_root']='/data/bound/child'
    with pytest.raises(ValueError):module.prepare(c,{})
    assert not Path(c['output']).exists()


@pytest.mark.parametrize('fault',['batch','hf','macro','runner','input','endpoint','source','extra',
    'layout','budget','admission','parent-proof','input-proof','symlink'])
def test_rehashed_mutations_fail_independent_reconstruction(tmp_path,monkeypatch,fault):
    c=fixture(tmp_path,monkeypatch);module.prepare(c,{'prepare':True})
    root=Path(c['output']);manifest=root/'PREPARED.json';m=module.read(manifest)
    if fault=='batch':
        p=root/'service-a.json';v=module.read(p);v['policy']['max_batch_queries']=4;write(p,v)
    elif fault in ('hf','macro'):
        p=root/f'parity-{fault}.json';v=module.read(p);v['macro_client']['rollout_queries']=4;write(p,v)
    elif fault=='runner':(root/'run.py').write_text('raise SystemExit(0)\n')
    elif fault=='input':write(root/'parity-inputs/inputs.json',{'fake':True})
    elif fault=='endpoint':write(root/'initial/endpoint-commit.json',{'updates':288})
    elif fault=='source':(root/'source/example.py').write_text('changed')
    elif fault=='extra':(root/'unrequested.py').write_text('pass')
    elif fault=='layout':m['layout']['directories'].remove('hf-process')
    elif fault=='budget':m['training_devices']=[0,1,2,3,4,5]
    elif fault=='admission':m['numerical_admission']=True
    elif fault=='parent-proof':write(root/'PARITY-PARENT-AUDIT.json',{'fixture':'wrong-parent'})
    elif fault=='input-proof':write(root/'PARITY-INPUT-AUDIT.json',{'fixture':'wrong-input'})
    else:
        p=root/'source/example.py';p.unlink();p.symlink_to(Path(c['parent']['output'])/'source/example.py')
    # A corrupted producer can regenerate all hashes; independent expectations
    # must still reject its altered policy, topology, code or collection scope.
    m['files']={str(p.relative_to(root)):module.file_hash(p) for p in root.rglob('*') if p.is_file() and p!=manifest}
    write(manifest,m)
    with pytest.raises(ValueError):module.audit(dict(c,manifest_sha256=module.file_hash(manifest)),{})
