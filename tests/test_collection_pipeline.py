import json
from pathlib import Path

import pytest

from tools import collect_training_artifacts as collect
from tools import export_training_trajectory as trajectory
from tools import run_operations as ops


def fixture(tmp_path):
    collection = tmp_path/'collected'; collection.mkdir()
    files = {}
    def put(name, value):
        p = collection/name; p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value)); files[name] = collect.sha(p)
    put('config.json', dict(physical_devices=[4,5,6,7], global_batch=1))
    put('service-config.json', {}); put('PREPARED.json', {})
    for rank in range(4):
        put(f'run/rank-{rank}/update-0001.json', dict(update=1, real_queries=1,
            preparation_seconds=1, actor_seconds=2, total_seconds=3, metrics={},
            rollout=dict(update=0, behavior_policy_id='b', fixed_reference_policy_id='r',
                queries=[dict(request_id='q', behavior_record={'policy_id':'b'},
                              reference_record={'policy_id':'r'})])))
    archive = collection.with_suffix('.tar'); archive.write_bytes(b'archive fixture')
    launch = dict(container_id='a'*64,image='sha256:'+'b'*64,remote_root='/data/run',
        config_sha256=files['config.json'],service_config_sha256=files['service-config.json'],
        manifest_sha256=files['PREPARED.json'])
    lp = tmp_path/'launch.json'; lp.write_text(json.dumps(launch))
    collection.with_suffix('.transport.json').write_text(json.dumps(dict(launch,
        status='transport_complete_independent_audit_required',state=dict(Status='exited',ExitCode=0),
        files=files,bad_pins=[],bad_package=[],launch_binding_mismatches=[],archive_sha256=collect.sha(archive))))
    local = tmp_path/'local.json'
    local.write_text(json.dumps(dict(schema='jev-local-macros-v1',macros=dict(
        WORKSPACE=str(Path.cwd()),COLLECTION_MODE='adopt',TRAIN_HOST='npu',COLLECTION_LAUNCH=str(lp),
        COLLECTION_LAUNCH_SHA256=collect.sha(lp),TRAJECTORY_COLLECTION=str(collection),
        COLLECTION_OUTPUT=str(tmp_path/'publication'),COLLECTION_LOCK=str(tmp_path/'collect.lock'),
        COLLECT_TRAJECTORY_PIPELINE_OUTPUT=str(tmp_path/'workflow'),TRAJECTORY_OUTPUT=str(tmp_path/'trajectory'),
        TRAJECTORY_TRANSPORT='/wrong-old-binding',TRAJECTORY_TRANSPORT_SHA256='wrong'))))
    return local, collection


def test_actual_adopt_to_trajectory_and_reuse(tmp_path, monkeypatch):
    local, collection = fixture(tmp_path)
    for m in (ops, collect, trajectory): monkeypatch.setattr(m,'check_paths',lambda *a:[])
    monkeypatch.setattr(collect,'collect',lambda *a,**k:pytest.fail('no remote'))
    template='configs/portable/operations/collect-trajectory-pipeline.json'
    original=local.read_bytes()
    assert ops.run(template,local)['completed']==['collect','trajectory']
    assert ops.run(template,local)['status']=='already_complete'
    assert local.read_bytes()==original
    assert json.loads((tmp_path/'trajectory/trajectory.jsonl').read_text())['request_id']=='q'
    bound=json.loads((tmp_path/'workflow/trajectory.macros.json').read_text())['macros']
    assert bound['TRAJECTORY_TRANSPORT_SHA256']==collect.sha(collection.with_suffix('.transport.json'))
    (collection/'run/rank-0/update-0001.json').write_text('{}')
    with pytest.raises(ValueError,match='identity changed'):ops.run(template,local)


def test_dependency_and_schema_required():
    config=json.loads(Path('configs/portable/operations/collect-trajectory-pipeline.json').read_text())
    config['steps'][1]['depends_on']=[]
    with pytest.raises(ValueError,match='direct'):ops.ordered_steps(config)
    config['schema']='local-operations-v2'
    with pytest.raises(ValueError,match='v3'):ops.ordered_steps(config)


def test_mutated_upstream_publication_rejected(tmp_path,monkeypatch):
    local,_=fixture(tmp_path)
    monkeypatch.setattr(collect,'check_paths',lambda *a:[])
    collect.run('configs/portable/operations/training-collection.json',local)
    complete=tmp_path/'publication/COMPLETE.json';digest=collect.sha(complete)
    complete.write_text('{}')
    with pytest.raises(ValueError,match='changed before binding'):
        ops.child_bindings({'collection_from':'collect'},local,{'collect':(str(complete),digest)})
