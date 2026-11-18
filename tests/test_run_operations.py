import json

import pytest

from tools import run_operations as ops


def fixture(tmp_path):
    child = tmp_path / 'child.json'; child.write_text('{"batch":128}')
    local = tmp_path / 'local.json'; local.write_text(json.dumps({'schema':'jev-local-macros-v1','macros':{}}))
    config = dict(schema='local-operations-v1', output_directory=str(tmp_path / 'workflow'), steps=[
        dict(id='first', kind='materialize', config=str(child), output=str(tmp_path/'first.json'), depends_on=[]),
        dict(id='second', kind='materialize', config=str(child), output=str(tmp_path/'second.json'), depends_on=['first'])])
    template = tmp_path / 'workflow.json'; template.write_text(json.dumps(config))
    return template, local, config


@pytest.mark.parametrize('mutation', ['cycle','missing','duplicate','shell'])
def test_invalid_dag_rejected_before_writes(tmp_path, mutation):
    template, local, config = fixture(tmp_path)
    if mutation=='cycle': config['steps'][0]['depends_on']=['second']
    elif mutation=='missing': config['steps'][1]['depends_on']=['absent']
    elif mutation=='duplicate': config['steps'][1]['id']='first'
    else: config['steps'][0]['kind']='shell'
    template.write_text(json.dumps(config))
    with pytest.raises(ValueError): ops.run(template, local)
    assert not (tmp_path/'workflow').exists()


def test_real_materialization_and_idempotence(tmp_path, monkeypatch):
    from tools import configure
    template, local, config=fixture(tmp_path)
    monkeypatch.setattr(ops,'check_paths',lambda *a:[])
    monkeypatch.setattr(configure,'check_paths',lambda *a:[])
    assert ops.run(template,local)==dict(status='complete',completed=['first','second'])
    assert json.loads((tmp_path/'second.json').read_text())=={'batch':128}
    monkeypatch.setattr(ops,'execute',lambda *a:pytest.fail('duplicate execution'))
    assert ops.run(template,local)['status']=='already_complete'
    (tmp_path/'first.json').write_text('{}')
    with pytest.raises(ValueError,match='Source changed'):ops.run(template,local)


def test_failure_blocks_downstream_and_retry(tmp_path, monkeypatch):
    template, local, _=fixture(tmp_path)
    monkeypatch.setattr(ops,'check_paths',lambda *a:[])
    calls=[]
    def fail(step,*args):
        calls.append(step['id']);raise RuntimeError('first failed')
    monkeypatch.setattr(ops,'execute',fail)
    with pytest.raises(RuntimeError):ops.run(template,local)
    assert calls==['first']
    assert not (tmp_path/'workflow/COMPLETE.json').exists()
    with pytest.raises(ValueError,match='Partial workflow'):ops.run(template,local)
    assert calls==['first']


def test_macro_error_in_second_step_prevents_first_side_effect(tmp_path,monkeypatch):
    template,local,config=fixture(tmp_path)
    missing=tmp_path/'missing.json';missing.write_text('{"path":"${ABSENT}"}')
    config['steps'][1]['config']=str(missing);template.write_text(json.dumps(config))
    monkeypatch.setattr(ops,'execute',lambda *a:pytest.fail('should prevalidate'))
    with pytest.raises(ValueError,match='Missing local macro'):ops.run(template,local)
    assert not (tmp_path/'workflow').exists()


def test_input_changes_during_step_stop_workflow(tmp_path,monkeypatch):
    template,local,_=fixture(tmp_path)
    monkeypatch.setattr(ops,'check_paths',lambda *a:[])
    def mutate(*args):
        local.write_text('{}');return {'status':'complete'},[]
    monkeypatch.setattr(ops,'execute',mutate)
    with pytest.raises(ValueError,match='Source changed'):ops.run(template,local)
    assert not (tmp_path/'workflow/COMPLETE.json').exists()


def test_dynamic_binding_requires_declared_signal_dependency(tmp_path):
    _, _, config = fixture(tmp_path)
    config['schema'] = 'local-operations-v2'
    config['steps'][1].update(kind='probability_errors', signal_from='first')
    del config['steps'][1]['output']
    with pytest.raises(ValueError, match='signal_report dependency'):
        ops.ordered_steps(config)


def test_dynamic_binding_overrides_stale_macro_and_rejects_changed_publication(tmp_path):
    local = tmp_path / 'local.json'
    local.write_text(json.dumps({'schema':'jev-local-macros-v1', 'macros':
        {'SIGNAL_PUBLICATION':'stale', 'SIGNAL_PUBLICATION_SHA256':'stale'}}))
    publication = tmp_path / 'COMPLETE.json'; publication.write_text('{}')
    before = local.read_bytes()
    registry = {'signal': (str(publication), ops.sha(publication))}
    value = ops.child_bindings({'signal_from':'signal'}, local, registry)
    assert value['macros']['SIGNAL_PUBLICATION'] == str(publication)
    assert value['macros']['SIGNAL_PUBLICATION_SHA256'] == ops.sha(publication)
    assert local.read_bytes() == before
    publication.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='Upstream publication changed'):
        ops.child_bindings({'signal_from':'signal'}, local, registry)


def test_v2_handoff_uses_completed_artifact_and_missing_macros_preflight(tmp_path, monkeypatch):
    local=tmp_path/'local.json'
    local.write_text(json.dumps({'schema':'jev-local-macros-v1','macros':{}}))
    signal=tmp_path/'signal.json'; signal.write_text('{}')
    errors=tmp_path/'errors.json'
    errors.write_text(json.dumps({'path':'${SIGNAL_PUBLICATION}', 'hash':'${SIGNAL_PUBLICATION_SHA256}'}))
    config=dict(schema='local-operations-v2',output_directory=str(tmp_path/'workflow'),steps=[
        dict(id='signal',kind='signal_report',config=str(signal),depends_on=[]),
        dict(id='errors',kind='probability_errors',config=str(errors),depends_on=['signal'],signal_from='signal')])
    template=tmp_path/'pipeline.json';template.write_text(json.dumps(config))
    monkeypatch.setattr(ops,'check_paths',lambda *a:[])
    calls=[]
    publication=tmp_path/'signal-COMPLETE.json'
    def execute(step,binding,root):
        calls.append(step['id'])
        if step['id']=='signal':
            publication.write_text(json.dumps(dict(schema='committed-signal-report-publication-v1',sources={},outputs={})))
            return dict(status='complete',manifest=str(publication)),[publication]
        from multimodal_jev.local_config import load_config
        resolved,_=load_config(step['config'],binding)
        assert resolved==dict(path=str(publication),hash=ops.sha(publication))
        return dict(status='complete'),[]
    monkeypatch.setattr(ops,'execute',execute)
    assert ops.run(template,local)['completed']==['signal','errors']
    assert calls==['signal','errors']


def trajectory_publication(tmp_path):
    root=tmp_path/'trajectory';root.mkdir()
    source=tmp_path/'input.json';source.write_text('{}')
    for name in ('trajectory.jsonl','updates.json'):
        (root/name).write_text('[]\n')
    publication=root/'COMPLETE.json'
    publication.write_text(json.dumps(dict(schema='recorded-training-trajectory-publication-v1',
        sources={str(source):ops.sha(source)},outputs={name:ops.sha(root/name) for name in ('trajectory.jsonl','updates.json')})))
    return publication,source


@pytest.mark.parametrize('fault',['input','output','missing_output','schema'])
def test_child_publication_not_accepted_by_current_hash_alone(tmp_path,fault):
    publication,source=trajectory_publication(tmp_path)
    assert ops.verified_publication('training_trajectory',publication)
    if fault=='input':source.write_text('changed')
    elif fault=='output':(publication.parent/'trajectory.jsonl').write_text('changed')
    else:
        value=json.loads(publication.read_text())
        if fault=='schema':value['schema']='wrong'
        else:del value['outputs']['updates.json']
        publication.write_text(json.dumps(value))
    with pytest.raises(ValueError):ops.verified_publication('training_trajectory',publication)


def test_workflow_reuse_checks_child_input_dependencies(tmp_path,monkeypatch):
    publication,source=trajectory_publication(tmp_path)
    child=tmp_path/'child.json';child.write_text('{}')
    local=tmp_path/'local.json';local.write_text(json.dumps({'schema':'jev-local-macros-v1','macros':{}}))
    template=tmp_path/'pipeline.json'
    template.write_text(json.dumps(dict(schema='local-operations-v2',output_directory=str(tmp_path/'workflow'),
        steps=[dict(id='trajectory',kind='training_trajectory',config=str(child),depends_on=[])])))
    monkeypatch.setattr(ops,'check_paths',lambda *a:[])
    monkeypatch.setattr(ops,'execute',lambda *a:(dict(status='complete'),[publication]))
    assert ops.run(template,local)['status']=='complete'
    source.write_text('changed')
    with pytest.raises(ValueError,match='Source changed'):ops.run(template,local)
