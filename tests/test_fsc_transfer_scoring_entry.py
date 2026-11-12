import importlib.util
from pathlib import Path
import sys
import pytest
from multimodal_jev.baseline import atomic_json, file_hash

TOOLS = Path(__file__).resolve().parents[1]/'tools/jev'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('fsc_scoring_entry', TOOLS/'score_fsc_span_transfer.py')
entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)


def fixture(tmp_path):
    task = tmp_path/'task.json'; population = tmp_path/'population.json'; selection = tmp_path/'selection.json'
    atomic_json(task, {}); atomic_json(population, {'records': [], 'split': 'test'})
    checkpoints = [dict(key=f'step{i}', adapter_id=f'adapter{i}', root=str(tmp_path/f'gate{i}')) for i in range(3)]
    atomic_json(selection, dict(checkpoints=[{k: e[k] for k in ('key', 'adapter_id')} for e in checkpoints],
                               target_feedback_used=False))
    protocol = dict(schema='fsc-span-transfer-evaluation-v1',
        selection_rule='source_only_frozen_before_target_inference', target_feedback_used_for_selection=False,
        binding_seed=317, bootstrap_seed=941, bootstrap_replicates=2000, base_id='base', checkpoints=checkpoints,
        task=str(task), task_sha256=file_hash(task), population=str(population), population_sha256=file_hash(population),
        source_selection=str(selection), source_selection_sha256=file_hash(selection))
    path = tmp_path/'protocol.json'; atomic_json(path, protocol)
    return path, protocol


def test_incomplete_matrix_never_reads_target_labels(tmp_path, monkeypatch):
    path, protocol = fixture(tmp_path); visited = []
    def gate(e, *args):
        visited.append(e['key'])
        if e['key'] == 'step1': raise ValueError('not complete')
        return {'records': [], 'audit': {}}, {}
    monkeypatch.setattr(entry, 'authenticate_gate', gate)
    monkeypatch.setattr(entry, 'scoring_targets', lambda *a: pytest.fail('targets read before complete matrix'))
    with pytest.raises(ValueError, match='not complete'): entry.run(path, file_hash(path))
    assert visited == ['step0', 'step1']


@pytest.mark.parametrize('bad', ['hash', 'selection', 'feedback', 'two_points', 'duplicate'])
def test_protocol_rejections_precede_inference_and_labels(tmp_path, monkeypatch, bad):
    path, protocol = fixture(tmp_path)
    if bad == 'selection': protocol['source_selection_sha256'] = 'wrong'
    if bad == 'feedback': protocol['target_feedback_used_for_selection'] = True
    if bad == 'two_points': protocol['checkpoints'].pop()
    if bad == 'duplicate': protocol['checkpoints'][1]['root'] = protocol['checkpoints'][0]['root']
    atomic_json(path, protocol)
    monkeypatch.setattr(entry, 'authenticate_gate', lambda *a: pytest.fail('gate called despite invalid protocol'))
    with pytest.raises(ValueError): entry.run(path, 'wrong' if bad == 'hash' else file_hash(path))


def test_gate_requires_protocol_registered_in_launched_config(tmp_path, monkeypatch):
    path, protocol = fixture(tmp_path); gate = tmp_path/'gate0'; gate.mkdir()
    atomic_json(gate/'config.json', {'pins': {}}); atomic_json(gate/'source-manifest.json', {})
    protocol.update(image='image', runtime_profile='profile')
    atomic_json(gate/'launch.json', dict(image='image', physical_devices=[4],
        config_sha256=file_hash(gate/'config.json'), source_manifest_sha256=file_hash(gate/'source-manifest.json')))
    monkeypatch.setattr(entry, 'inspect', lambda *a: pytest.fail('late registration reached execution check'))
    with pytest.raises(ValueError, match='prospective'):
        entry.authenticate_gate(protocol['checkpoints'][0], path, file_hash(path), protocol, {}, {})
