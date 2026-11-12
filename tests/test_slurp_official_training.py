from dataclasses import asdict
import json
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.slurp_training import content_hash
from multimodal_jev.slurp_official_training import publish, load, example, source_population
from multimodal_jev.task_schema import Candidate, DecisionSchema


def fixture(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    review_paths = {}
    for j, role in enumerate(('learning', 'selection')):
        requests = []; targets = []; packet = []
        for i in range(3):
            audio = tmp_path/f'{role}-{i}.wav'
            sf.write(audio, np.sin(np.arange(1600)*(.01+.003*(j*3+i))).astype('float32')*.2, 16000)
            request = dict(id='slurp:'+audio.name, official_split='train', role=role,
                source_group=f'{role}-group-{i}', audio_sha256=file_hash(audio), input={'audio': str(audio)})
            target = dict(id=request['id'], request_sha256=content_hash(request), annotation_status='available',
                          native_target=dict(scenario='alarm', action='set', entities=[]),
                          semantic_review_status='pending', optimization_allowed=False)
            requests.append(request); targets.append(target); packet.append(dict(request=request, target=target))
        for suffix, data in [('requests', requests), ('targets', targets)]:
            (source/f'{role}.{suffix}.jsonl').write_bytes(b''.join(map(canonical, data)))
        packet_path = tmp_path/f'{role}-packet.json'; packet_path.write_bytes(canonical(packet))
        if role == 'learning':
            audit = dict(reviewed={'0': 'fixture', '1': 'fixture'}, pending={'2': 'ambiguous fixture'})
        else:
            audit = dict(records=[dict(index=i, id=v['request']['id'], source_group=v['request']['source_group'],
                record_sha256=content_hash(v), status='pending' if i == 2 else 'reference_consistent') for i, v in enumerate(packet)])
        audit['packet_sha256'] = file_hash(packet_path)
        audit_path = tmp_path/f'{role}-audit.json'; audit_path.write_bytes(canonical(audit))
        review_paths[role] = [str(packet_path), str(audit_path)]
    manifest = dict(schema='slurp-clean-source-bundles-v1', counts={'learning': 3, 'selection': 3},
                    files={p.name: file_hash(p) for p in source.iterdir()})
    (source/'MANIFEST.json').write_bytes(canonical(manifest))
    receipt = tmp_path/'receipt.json'
    receipt.write_bytes(canonical(dict(release=str(source), manifest_sha256=file_hash(source/'MANIFEST.json'),
        all_native_targets_reconstructed_independently=True, cross_role_source_group_or_audio_overlap=0)))
    schema = DecisionSchema('test-intent', 'choice', 'Which requested task?',
        tuple(Candidate(i, i) for i in ('alarm_set', 'calendar_set', 'alarm_query')), False)
    task = dict(kind='intent', schema=asdict(schema), candidate_pairs={'alarm_set': ['alarm', 'set'],
        'calendar_set': ['calendar', 'set'], 'alarm_query': ['alarm', 'query']},
        thinking={'requirements': ['Use only the audio and supplied questions.']})
    tasks = tmp_path/'tasks.json'; tasks.write_bytes(canonical({'intent': task}))
    paths = [source/'MANIFEST.json', receipt, tasks]+[Path(p) for pair in review_paths.values() for p in pair]
    config = dict(source=str(source), receipt=str(receipt), tasks=str(tasks), reviews=review_paths,
                  pins={str(p): file_hash(p) for p in paths})
    return config, tmp_path/'release'


def test_full_release_masks_pending_and_keeps_roles_and_semantics(tmp_path):
    config, output = fixture(tmp_path)
    before = Path(config['source'],'learning.targets.jsonl').read_bytes()
    report = publish(config, output)
    assert report['counts']['learning']['recordings'] == 2
    assert report['counts']['selection']['recordings'] == 2
    assert report['quarantine'] == {'known_pending_source': 2}
    assert report['individual_semantic_review'] is False
    assert Path(config['source'],'learning.targets.jsonl').read_bytes() == before
    data, task, _ = load(output, file_hash(output/'MANIFEST.json'), 'learning', purpose='optimize')
    a, target, _ = example(data[0], task, 'structured', 'shift1')
    assert len(a.questions) == 3 and target['span_weights'] == [.1, .1, 1.]
    assert [q.values[y] for q, y in zip(a.questions, target['targets'])] == ['alarm', 'set', 'alarm_set']
    changed = dict(data[0], intent='calendar_set', scenario='calendar')
    b, other, _ = example(changed, task, 'structured', 'shift1')
    assert a.model_input() == b.model_input() and target['targets'] != other['targets']
    for view in ('direct', 'scenario', 'action'):
        req, gold, _ = example(data[0], task, view)
        assert len(req.questions) == 1 and gold['span_weights'] == [1.]
    with pytest.raises(ValueError, match='selection cannot optimize'):
        load(output, file_hash(output/'MANIFEST.json'), 'selection', purpose='optimize')
    with pytest.raises(FileExistsError):
        publish(config, output)


def test_audio_failure_is_explicit_quarantine(tmp_path):
    config, output = fixture(tmp_path)
    (tmp_path/'learning-0.wav').write_bytes(b'corrupted')
    report = publish(config, output)
    assert report['counts']['learning']['recordings'] == 1
    assert report['quarantine']['invalid_audio_asset'] == 1
    data, _, _ = load(output, file_hash(output/'MANIFEST.json'), 'learning', purpose='optimize')
    assert len(data) == 1


def test_cross_role_source_leakage_is_rejected_even_with_rehashed_source(tmp_path):
    config, output = fixture(tmp_path)
    source = Path(config['source'])
    rp = source/'selection.requests.jsonl'; tp = source/'selection.targets.jsonl'
    requests = [json.loads(x) for x in rp.read_text().splitlines()]
    targets = [json.loads(x) for x in tp.read_text().splitlines()]
    requests[0]['source_group'] = 'learning-group-0'
    targets[0]['request_sha256'] = content_hash(requests[0])
    rp.write_bytes(b''.join(map(canonical, requests))); tp.write_bytes(b''.join(map(canonical, targets)))
    manifest = json.loads((source/'MANIFEST.json').read_text())
    for path in (rp, tp):
        manifest['files'][path.name] = file_hash(path)
    (source/'MANIFEST.json').write_bytes(canonical(manifest))
    receipt = json.loads(Path(config['receipt']).read_text())
    receipt['manifest_sha256'] = file_hash(source/'MANIFEST.json')
    Path(config['receipt']).write_bytes(canonical(receipt))
    for path in config['pins']:
        config['pins'][path] = file_hash(path)
    with pytest.raises(ValueError, match='cross-role'):
        publish(config, output)
    assert not output.exists()


def test_independent_verifier_reconstructs_targets_and_counts(tmp_path):
    path = Path(__file__).resolve().parents[1]/'tools/jev/verify_slurp_official_intent.py'
    spec = importlib.util.spec_from_file_location('independent_official_verifier', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    config, output = fixture(tmp_path); publish(config, output)
    result = module.verify(output, file_hash(output/'MANIFEST.json'))
    assert result['passed'] and result['counts']['learning']['recordings'] == 2
    # A manifest alone cannot silently remove a difficult source from the denominator.
    p = output/'learning.jsonl'
    p.write_text(p.read_text().splitlines()[0]+'\n')
    m = json.loads((output/'MANIFEST.json').read_text()); m['files'][p.name] = file_hash(p)
    m['counts']['learning']['recordings'] = 1
    (output/'MANIFEST.json').write_bytes(canonical(m))
    with pytest.raises(ValueError, match='complete population'):
        module.verify(output, file_hash(output/'MANIFEST.json'))


@pytest.mark.parametrize('what', ['source', 'review', 'target', 'role'])
def test_release_refuses_stale_dependencies_or_modified_rows(tmp_path, what):
    config, output = fixture(tmp_path)
    if what in ('source', 'review'):
        path = Path(config['source'])/'learning.targets.jsonl' if what == 'source' else Path(config['reviews']['learning'][1])
        path.write_text('{}\n')
        with pytest.raises(ValueError):
            publish(config, output)
        assert not output.exists()
    else:
        publish(config, output)
        p = output/'learning.jsonl'; values = [json.loads(line) for line in p.read_text().splitlines()]
        values[0][what if what == 'role' else 'intent'] = 'selection' if what == 'role' else 'calendar_set'
        p.write_bytes(b''.join(map(canonical, values)))
        # Even if somebody rewrites the publication manifest, source rederivation refuses the change.
        m = json.loads((output/'MANIFEST.json').read_text()); m['files'][p.name] = file_hash(p)
        (output/'MANIFEST.json').write_bytes(canonical(m))
        with pytest.raises(ValueError, match='native labels'):
            load(output, file_hash(output/'MANIFEST.json'), 'learning', purpose='optimize')
