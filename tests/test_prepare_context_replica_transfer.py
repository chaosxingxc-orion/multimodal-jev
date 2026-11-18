import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tools.reports import prepare_context_replica_transfer as m
from test_prepare_context_replica_selection import setup as selection_setup, write
from test_context_transfer_macro_evaluation import source_config


def setup(tmp_path, monkeypatch):
    from tools.reports import prepare_context_replica_selection as selection
    from tools.reports.evaluate_context_transfer_macro import source_binding, read_bound
    c, parent = selection_setup(tmp_path, monkeypatch)
    _, proof, parent, manifest, training, point, bundle, _, _ = selection.inputs(c,{})
    c.update(dataset='FSC_native31', source_admission=source_config(tmp_path))
    c['source_admission']['source_curve_config'] = {'source_plan':c['source_plan']}
    source = read_bound(c['source_admission']['source_evaluation'])
    point = dict(point, step=0, policy_id='p0')
    admission = dict(status='source_role_runtime_and_weights_admitted', binding=source_binding(c['source_admission']),
        source=source, curve={}, audits={}, points=[dict(step=p['step'], result=read_bound(p['result'])) for p in source['points']],
        source_evaluation=c['source_admission']['source_evaluation'], source_curve_config=c['source_admission']['source_curve_config'], point=point)
    for key in ('portable_full_inputs', 'portable_parity_inputs'):
        path = tmp_path/key
        write(path/'MANIFEST.json', {'fixture':key})
        c[key] = dict(root=str(path), manifest_sha256=m.file_hash(path/'MANIFEST.json'), cache_root='/data/target-cache/input')
    monkeypatch.setattr(m, 'inputs', lambda *a:(admission, proof, parent, manifest, training, point, bundle,
        dict(population=[{}]*3793, candidate_ids=list(range(31))), {'fixture':'panel'}, {'fixture':'coverage'}))
    monkeypatch.setattr(m, 'export_native_adapter', selection.export_native_adapter)
    monkeypatch.setattr(m, 'check_paths', lambda _:None)
    return c, parent


def test_new_package_reconstructs_and_preserves_model_sources(tmp_path, monkeypatch):
    c, parent = setup(tmp_path, monkeypatch)
    before = {str(p):m.file_hash(p) for p in parent.rglob('*') if p.is_file()}
    c['manifest_sha256'] = m.prepare(c,{})['manifest_sha256']
    root = Path(c['output'])
    manifest = m.read(root/'PREPARED.json')
    assert manifest['recordings'] == 3793 and manifest['actor_updates'] == 0
    assert manifest['policy_role'] == 'common_Stage1'
    assert before == {str(p):m.file_hash(p) for p in parent.rglob('*') if p.is_file()}
    assert (root/'RUNTIME-SOURCES.json').read_bytes() == (parent/'RUNTIME-SOURCES.json').read_bytes()
    assert m.read(root/'parity-macro.json')['macro_client']['journal_root'] == c['remote_root']+'/transfer-journal'
    for name in manifest['layout']['services'].values():
        assert m.read(root/name)['policy']['max_decisions'] == 31
        assert m.read(root/name)['policy']['policy_id'] == 'p0'
    assert m.audit(c,{})['status'] == 'replica_transfer_package_audited_not_executed'
    assert not list(root.parent.glob('transfer-reconstruct-*'))


@pytest.mark.parametrize('fault', ['source','width','runtime','wrapper','input','config','manifest'])
def test_rehashed_package_changes_cannot_pass_reconstruction(tmp_path, monkeypatch, fault):
    c, _ = setup(tmp_path, monkeypatch)
    m.prepare(c,{})
    root = Path(c['output'])
    manifest = m.read(root/'PREPARED.json')
    if fault == 'source': write(root/'source-evidence/score-0000.json', {})
    elif fault == 'width':
        v = m.read(root/'service-a.json');v['policy']['max_decisions']=60;write(root/'service-a.json',v)
    elif fault == 'runtime': (root/'source/example.py').write_text('changed')
    elif fault == 'wrapper': (root/'transfer-client.py').write_text('pass')
    elif fault == 'input': write(root/'full-inputs/MANIFEST.json', {})
    elif fault == 'config':
        v = m.read(root/'parity-macro.json');v['macro_queries']=32;write(root/'parity-macro.json',v)
    else: manifest['recordings']=1
    manifest['files'] = {str(p.relative_to(root)):m.file_hash(p) for p in root.rglob('*') if p.is_file() and p != root/'PREPARED.json'}
    write(root/'PREPARED.json', manifest);c['manifest_sha256']=m.file_hash(root/'PREPARED.json')
    with pytest.raises(ValueError): m.audit(c,{})


@pytest.mark.parametrize('code', [0, 6])
def test_generated_runner_owns_hf_then_replicas_without_training(tmp_path, monkeypatch, code):
    from multimodal_jev import local_checkpoint, local_runtime_storage
    from tools.jev.context_process_supervisor import stop_owned
    monkeypatch.setattr(local_checkpoint, 'check_paths', lambda _:None)
    monkeypatch.setattr(local_runtime_storage, 'require_training_storage', lambda _:None)
    manifest = dict(training_devices=[4,5], inference_services=[dict(name='a',device=6),dict(name='b',device=7)])
    write(tmp_path/'parity-hf.json', dict(dataset='FSC_native31',expected_population=3793,process_device_scope={'visible_ordinal':'0'}))
    for name in ('a','b'): write(tmp_path/f'service-{name}.json', {'socket_path':str(tmp_path/name/'socket')})
    (tmp_path/'transfer-client.py').write_text('import json,os\nfrom pathlib import Path\nr=Path(__file__).parent\n'
        +"assert os.environ['ASCEND_RT_VISIBLE_DEVICES']=='0'\n(r/'hf').mkdir()\n"
        +"(r/'hf/terminal.json').write_text(json.dumps({'exit_code':"+str(code)+"}))\n"
        +"(r/'hf/reference.json').write_text('{}')\nraise SystemExit("+str(code)+')\n')
    calls=[]
    def supervise(**kw):
        import os
        with pytest.raises(ProcessLookupError): os.kill(m.read(tmp_path/'hf-process/terminal.json')['pid'],0)
        calls.append(kw);return {'passed':True}
    monkeypatch.setitem(sys.modules,'context_process_supervisor',SimpleNamespace(supervise_services=supervise,stop_owned=stop_owned))
    (tmp_path/'cache').mkdir()
    generated=m.runner(manifest).replace("Path('/jev-tmp')",'Path('+repr(str(tmp_path/'cache'))+')')
    with pytest.raises(SystemExit) as e: exec(compile(generated,'<transfer-runner>','exec'), {'__file__':str(tmp_path/'run.py')})
    assert e.value.code==code and len(calls)==int(code==0)
    if calls:
        assert 'transfer-client.py' in calls[0]['actor_command'][2]
        assert 'torchrun' not in str(calls)
        assert [s['env']['ASCEND_RT_VISIBLE_DEVICES'] for s in calls[0]['services']]==['2','3']


def test_parent_mismatch_rejected_before_target_access(monkeypatch):
    monkeypatch.setattr(m,'admit',lambda _: {'source_operations':{'training_package':{'parent':{'expected':True}}}})
    monkeypatch.setattr(m.client,'inputs',lambda _:pytest.fail('target accessed'))
    with pytest.raises(ValueError,match='parent'):m.inputs({'source_admission':{},'parent':{}},{})
