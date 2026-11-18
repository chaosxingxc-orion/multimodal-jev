"""The composed controller's own guarantees, independent of the differential run.

The differential audit proves the composed path equals the frozen rewrite on one
frozen configuration. These tests pin the properties that make that equality
mean something when the configuration changes: the injected functions keep the
frozen code objects, the argv builder matches the two-anchor rewrite in both
producer modes, the lifecycle releases its lease before collecting and aborts
the queue on failure, and the entry point still refuses dispatch.
"""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from multimodal_jev import local_checkpoint
from tools.reports import context_cached_storage_operations as storage
from tools.reports import context_composed_pipeline as composed
from tools.reports import context_replica_training_operations as shared
from tools.reports.audit_context_composed_pipeline import Queue, Recorder

REMOTE = '/data/run'


def shared_len(value):
    return len(value)


def sample(value, flag=False):
    if flag:
        return value
    return shared_len(value)


def fixture(tmp_path, *, producer=False):
    """One admission-shaped configuration and the package its mounts describe."""
    reference = dict(root='/data/reference', manifest_sha256='c'*64)
    config = dict(source_root=REMOTE + '/source',
                  source_policy=REMOTE + '/initial/policy',
                  encoded_audio_cache='/data/cache/encoded',
                  audio_mirror_root='/data/mirror',
                  pins={REMOTE + '/config.json': '0'*64})
    if not producer:
        config['reference_dataset'] = reference
        config['pins'] = dict(config['pins'], **{reference['root'] + '/MANIFEST.json':
                                                 reference['manifest_sha256']})
    m = dict(remote_root=REMOTE, training_devices=[4, 5],
             inference_services=[dict(name='a', device=6), dict(name='b', device=7)],
             layout=dict(configs=['config.json', 'PREPARED.json', 'run.py'],
                         services={'a': 'service-a.json', 'b': 'service-b.json'},
                         directories=[]))
    package = tmp_path/'package'
    (package/'source/src/multimodal_jev').mkdir(parents=True)
    source = Path(__file__).resolve().parents[1]/'src/multimodal_jev/local_runtime_storage.py'
    (package/'source/src/multimodal_jev/local_runtime_storage.py').write_bytes(
        source.read_bytes())
    (package/'config.json').write_text(json.dumps(config))
    (package/'PREPARED.json').write_text(json.dumps({'files': {}}))
    (package/'run.py').write_text('pass\n')
    for name in ('a', 'b'):
        (package/f'service-{name}.json').write_text(json.dumps({'name': name}))
    (tmp_path/'lease.lock').write_text('lease\n')
    (tmp_path/'ops').mkdir()
    c = dict(tag='jev-composed-unit', host='npu', package=str(package),
             manifest_sha256='0'*64, shared_root='/shared/release',
             driver_root='/usr/local/Ascend/driver',
             operations_root=str(tmp_path/'ops'), lease=str(tmp_path/'lease.lock'))
    return c, m, config


def test_bind_keeps_the_frozen_code_object_and_refuses_bad_names():
    bound = composed.bind(sample, shared_len=lambda value: len(value) + 1)
    # Same bytecode, only the globals dict differs: a rename cannot silently
    # reroute the call, it either resolves here or fails here.
    assert bound.__code__ is sample.__code__
    assert bound('ab', False) == 3
    assert sample('ab', False) == 2
    # FunctionType carries neither defaults nor kwdefaults, so an omitted
    # argument raises instead of silently taking a different value. Every
    # frozen seam bind() is used on takes its arguments explicitly.
    with pytest.raises(TypeError):
        bound('ab')
    with pytest.raises(ValueError, match='cannot inject a local name: value'):
        composed.bind(sample, value=1)
    with pytest.raises(ValueError, match='frozen function does not reference: missing'):
        composed.bind(sample, missing=1)


def test_command_matches_the_two_anchor_rewrite_in_both_producer_modes(tmp_path):
    for producer in (True, False):
        c, m, config = fixture(tmp_path/f'producer-{producer}', producer=producer)
        expected = storage.command(c, m, config, producer=producer)
        assert composed.command(c, m, config, producer=producer) == expected
        cache = config['encoded_audio_cache']
        assert cache + ':' + cache + ':ro' in expected
        reference = '/data/reference:/data/reference:ro'
        assert (reference in expected) is (not producer)


def test_command_refuses_a_pin_outside_every_mount_or_a_moved_endpoint(tmp_path):
    c, m, config = fixture(tmp_path)
    with pytest.raises(ValueError, match='dependency outside explicit mounts'):
        composed.command(c, m, dict(config, pins={'/elsewhere/x': '0'*64}))
    with pytest.raises(ValueError, match='packaged current endpoint under run/initial'):
        composed.command(c, m, dict(config, source_policy='/data/other/policy'))


def test_lifecycle_holds_the_lease_and_collects_after_releasing_it(tmp_path):
    c, m, config = fixture(tmp_path, producer=True)
    recorder = Recorder(shared.IMAGE)
    recorder.argv = composed.command(c, m, config, producer=True)
    queue = Queue()
    launch = composed.stage_launcher(lambda _c: (m, config, {}), queue, 0, producer=True,
                                     authorization='selection_dispatch_allowed',
                                     effects=SimpleNamespace(**recorder.namespace()))
    result = launch(dict(c, selection_dispatch_allowed=True), {'case': 'unit'})
    assert result['exit_code'] == 0 and queue.failed is False
    assert result['status'] == 'selection_collected_independent_audit_required'
    assert queue.events == ['check', 'enter:0:jev-composed-unit', 'exit']
    # Collection reads already-written evidence, so it must come after the
    # queue slot and the device lease are released, not inside them.
    names = [call.get('name', call['kind']) for call in recorder.calls]
    assert names[-6:] == ['wait_exact', 'jev-composed-unit-wait.json', 'run',
                          'jev-composed-unit-inspect.json', 'collect',
                          'jev-composed-unit-collection.json']
    written = [call['name'] for call in recorder.calls if call['kind'] == 'write']
    assert written == ['jev-composed-unit-attempt.json', 'jev-composed-unit-launch.json',
                       'jev-composed-unit-start-response.json', 'jev-composed-unit-wait.json',
                       'jev-composed-unit-inspect.json', 'jev-composed-unit-collection.json']


def test_lifecycle_requires_authorization_before_touching_the_queue(tmp_path):
    c, m, config = fixture(tmp_path)
    recorder = Recorder(shared.IMAGE)
    queue = Queue()
    launch = composed.stage_launcher(lambda _c: (m, config, {}), queue, 0, producer=False,
                                     authorization='finite_training_dispatch_allowed',
                                     effects=SimpleNamespace(**recorder.namespace()))
    with pytest.raises(ValueError, match='explicit dispatch authorization required'):
        launch(c, {'case': 'unit'})
    assert queue.events == [] and recorder.calls == []


def test_failed_container_aborts_the_queue_and_keeps_the_collection(tmp_path):
    c, m, config = fixture(tmp_path)
    recorder = Recorder(shared.IMAGE)
    recorder.argv = composed.command(c, m, config, producer=False)
    effects = SimpleNamespace(**recorder.namespace())
    effects.wait_exact = lambda remote, container: 1
    queue = Queue()
    launch = composed.stage_launcher(lambda _c: (m, config, {}), queue, 0, producer=False,
                                     authorization='selection_dispatch_allowed',
                                     effects=effects)
    with pytest.raises(ValueError, match='NPU execution failed; collection preserved'):
        launch(dict(c, selection_dispatch_allowed=True), {'case': 'unit'})
    assert queue.failed is True
    # _launch aborts at once and run_stage aborts again on the way out; both are
    # recorded, and the collected evidence is never discarded on failure.
    assert queue.events.count('abort') == 2
    names = [call.get('name', call['kind']) for call in recorder.calls]
    assert names.index('collect') > names.index('jev-composed-unit-inspect.json')
    assert 'jev-composed-unit-collection.json' in names


def test_validate_delegates_to_the_frozen_admission_chain(monkeypatch):
    engine = importlib.import_module('tools.reports.context_audit_optimized_pipeline')
    seen = []

    def admission(c):
        seen.append(c)
        return {'selection': 'frozen-result'}

    monkeypatch.setattr(engine, 'validate', admission)
    anything = {'any': 'configuration'}
    assert composed.validate(anything) == {'selection': 'frozen-result'}
    assert seen == [anything]


def test_selection_operations_reuse_the_frozen_code_objects():
    point = importlib.import_module('tools.reports.context_replica_selection_operations')
    preparation = SimpleNamespace(prepare=lambda *a: None, audit=lambda *a: None)
    bound = composed.selection_operations(preparation)
    for name in ('validate', 'launch', 'audit', 'run'):
        assert getattr(bound, name).__code__ is getattr(point, name).__code__
    assert bound.validate.__globals__['preparation'] is preparation
    assert bound.audit.__globals__['preparation'] is preparation
    assert bound.launch.__globals__['validate'] is bound.validate
    assert bound.run.__globals__['launch'] is bound.launch
    assert bound.run.__globals__['audit'] is bound.audit


def test_selection_collector_binds_the_frozen_reconstructor(monkeypatch):
    curve = importlib.import_module('tools.reports.collect_context_replica_infonce_curve')
    plan = lambda c: {'step': c['step']}                      # noqa: E731 - injected seam
    reconstruct, collector = composed.selection_collector(curve, point='point',
                                                          reconstruct_plan=plan)
    assert reconstruct.__code__ is curve.reconstruct.__code__
    assert reconstruct.__globals__['point'] == 'point'
    assert reconstruct.__globals__['reconstruct_plan'] is plan
    calls = []
    monkeypatch.setattr(curve.publication, 'collect',
                        lambda c, p, reconstructor: (calls.append((c, p, reconstructor))
                                                     or {'status': 'collected'}))
    assert collector.collect({'curve': 1}, {'provenance': 2}) == {'status': 'collected'}
    assert calls == [({'curve': 1}, {'provenance': 2}, reconstruct)]


def test_selection_curve_publishes_every_registered_step(tmp_path, monkeypatch):
    monkeypatch.setattr(local_checkpoint, 'check_paths', lambda paths: None)
    steps = [0, 4]
    c = dict(output=str(tmp_path/'out'), prepared_root=str(tmp_path/'packages'),
             operations_root=str(tmp_path/'jobs'), remote_root=str(tmp_path/'remote'),
             tag_prefix='jev-unit', evaluation_workers=2, deployment=dict(any='thing'),
             preprepared_stage1_endpoint=dict(package='/immutable/endpoint',
                                              remote_root='/immutable/endpoint',
                                              manifest_sha256='b'*64))
    calls = []

    def prepare(spec, provenance):
        calls.append(('prepare', spec['source_step']))
        return {'manifest_sha256': 'a'*64}

    def audit(spec, provenance):
        calls.append(('audit', spec['source_step']))

    def operation(point, ops, provenance, queue, priority, *, producer=False,
                  authorization):
        step = ops['selection_package']['source_step']
        target = Path(ops['operations_root'])/'finalization'
        target.mkdir(parents=True)
        local_checkpoint.commit_json(target/'audit-config.json', {'step': step})
        assert producer is True and authorization == 'selection_dispatch_allowed'
        return dict(status='complete_replica_infonce_selection_audited', source_step=step,
                    runtime_provenance_verified=True)

    queue = Queue()
    cc, result = composed.selection_curve(
        c, {'selection_inputs': {}}, steps, {'p': 1}, queue,
        preparation=SimpleNamespace(prepare=prepare, audit=audit), point=None,
        curve=SimpleNamespace(collect=lambda config, provenance: dict(
            status='selection_curve_collected', points=len(config['evaluations']))),
        operation=operation)
    root = Path(c['output'])/'infonce'
    assert [entry['step'] for entry in cc['evaluations']] == steps
    assert cc['output'] == str(root/'curve')
    assert result == {'status': 'selection_curve_collected', 'points': 2}
    assert sorted(calls) == [('audit', 4), ('prepare', 0)]
    assert queue.events == ['check', 'check']
    assert sorted(path.name for path in root.iterdir()) == [
        'completed-0000.json', 'completed-0004.json', 'curve-config.json',
        'template.json', 'terminal.json']


def test_run_refuses_dispatch_in_both_declared_forms():
    with pytest.raises(ValueError, match='construction-only'):
        composed.run({'dispatch_allowed': True}, {'p': 1})
    with pytest.raises(ValueError, match='construction-only'):
        composed.run({'dispatch_allowed': False}, {'p': 1}, dispatch=True)
