import copy

import pytest

from multimodal_jev.context_candidate_task import compile_candidates
from multimodal_jev.context_decision_contract import render_context_request
from multimodal_jev.slurp_training import content_hash
from tools.reports.preflight_slurp_context_test_inputs import reconstruct


def fixture():
    definitions = dict(candidates=[dict(id='c'+str(i), definition='Event '+str(i))
                                  for i in range(60)],
                       requirements=['Use supplied definitions.'], semantics='exclusive_event')
    request = compile_candidates('/audio/test.flac', **definitions)
    system, text = render_context_request(request)
    row = dict(id='slurp:test.flac', audio=request.audio, audio_sha256='0'*64,
               source_group='group', request_sha256=content_hash(request.model_input()),
               rendered_sha256=content_hash(dict(system=system, text=text)))
    return row, definitions


def test_frozen_roster_reconstructs_complete_menu():
    row, definitions = fixture()
    request = reconstruct(row, definitions)
    assert [q.id for q in request.questions] == ['c'+str(i) for i in range(60)]
    assert request.audio == row['audio']


@pytest.mark.parametrize('fault', ['label', 'extra_field', 'wrong_id', 'request_hash',
    'render_hash', 'missing_candidate', 'reorder', 'semantic_change', 'rule_change'])
def test_changed_input_or_supervision_rejected(fault):
    row, definitions = copy.deepcopy(fixture())
    if fault == 'label':
        row['target'] = 'c0'
    elif fault == 'extra_field':
        row['policy_id'] = 'policy'
    elif fault == 'wrong_id':
        row['id'] = 'slurp:other.flac'
    elif fault == 'request_hash':
        row['request_sha256'] = 'wrong'
    elif fault == 'render_hash':
        row['rendered_sha256'] = 'wrong'
    elif fault == 'missing_candidate':
        definitions['candidates'].pop()
    elif fault == 'reorder':
        definitions['candidates'].reverse()
    elif fault == 'semantic_change':
        definitions['semantics'] = 'candidate_applicability'
    else:
        definitions['requirements'].append('New rule.')
    with pytest.raises(ValueError):
        reconstruct(row, definitions)


@pytest.mark.parametrize('fault', [None, 'tensor_corrupt', 'wrong_marker', 'missing_receipt',
                                  'extra_receipt', 'roster_change'])
def test_independent_verifier_checks_actual_cache_tensors(tmp_path, monkeypatch, fault):
    import json
    import torch
    from multimodal_jev.baseline import file_hash
    from multimodal_jev.cached_context_preparation import CachedContextPreparer
    from multimodal_jev.encoded_audio_cache import EncodedAudioCache, key_for
    from multimodal_jev.span_request_identity import tensor_identity
    from tools.reports import preflight_slurp_context_test_inputs as module

    source, definitions = fixture()
    cache = EncodedAudioCache(tmp_path / 'cache')
    prepare = CachedContextPreparer(lambda: None, cache, processor_sha256='1'*64,
        decision_token_id=151706, audio_token_id=151676, max_tokens=4096)
    request = reconstruct(source, definitions)
    identity = prepare.identity(request, source['audio_sha256'])
    ids = torch.tensor([[151676] + [151706]*60])
    if fault == 'wrong_marker':
        ids[0, 1] = 42
    inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
        input_features=torch.zeros((1, 4, 8), dtype=torch.bfloat16),
        input_features_mask=torch.ones((1, 8), dtype=torch.long))
    positions = torch.arange(1, 61)[None, :]
    path = cache.write(identity, inputs, positions, {})
    receipt = dict(id=source['id'], audio_sha256=source['audio_sha256'],
        request_sha256=source['request_sha256'], rendered_sha256=source['rendered_sha256'],
        cache_key=key_for(identity), cache_sha256=file_hash(path),
        tensors={k: tensor_identity(v) for k, v in dict(inputs, positions=positions).items()},
        positions=positions.tolist(), tokens=61)
    if fault == 'tensor_corrupt':
        payload = torch.load(path, weights_only=True)
        payload['inputs']['input_features'][0, 0, 0] = 1
        torch.save(payload, path)
        receipt['cache_sha256'] = file_hash(path)  # outer hash alone is insufficient
    if fault == 'roster_change':
        receipt['id'] = 'slurp:other.flac'
    records = [] if fault == 'missing_receipt' else [receipt]
    if fault == 'extra_receipt':
        records.append(receipt)
    (tmp_path / 'inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    manifest = dict(schema='slurp-clean-context-test-processor-v1', recordings=12991,
        decisions=60, context_limit=4096, cache_root=str(cache.root),
        input_manifest_sha256='manifest', receipts_sha256=file_hash(tmp_path / 'inputs.jsonl'),
        processor_sha256='1'*64, max_tokens=61)
    (tmp_path / 'MANIFEST.json').write_text(json.dumps(manifest))
    # Population and processor factory are substitutes; tensor storage/readback,
    # per-row identity, marker checks and strict receipt cardinality are real.
    monkeypatch.setattr(module, 'authenticate', lambda c: ([source], definitions))
    monkeypatch.setattr(module, 'runtime', lambda c, o: (cache, prepare, '1'*64))
    if fault:
        with pytest.raises(ValueError):
            module.verify(dict(input_manifest_sha256='manifest'), tmp_path)
    else:
        assert module.verify(dict(input_manifest_sha256='manifest'), tmp_path)['max_tokens'] == 61


def test_preflight_updates_progress_twice_and_commits_final(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import torch
    from multimodal_jev.cached_context_preparation import CachedContextPreparer
    from multimodal_jev.encoded_audio_cache import EncodedAudioCache
    from multimodal_jev import local_checkpoint, local_runtime_storage
    from tools.reports import preflight_slurp_context_test_inputs as module

    source, definitions = fixture()
    request = reconstruct(source, definitions)
    cache = EncodedAudioCache(tmp_path / 'cache')
    builder = CachedContextPreparer(lambda: None, cache, processor_sha256='1'*64,
        decision_token_id=151706, audio_token_id=151676, max_tokens=4096)
    ids = torch.tensor([[151676] + [151706]*60])
    inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
        input_features=torch.zeros((1, 4, 8), dtype=torch.bfloat16),
        input_features_mask=torch.ones((1, 8), dtype=torch.long))
    positions = torch.arange(1, 61)[None, :]
    cache.write(builder.identity(request, source['audio_sha256']), inputs, positions, {})
    class Prepared:
        identity = builder.identity
        def __call__(self, job):
            return SimpleNamespace(inputs=inputs, positions=positions)
    # Repeated synthetic records exercise progress boundaries, not roster admission.
    monkeypatch.setattr(module, 'authenticate', lambda c: ([source]*513, definitions))
    monkeypatch.setattr(module, 'runtime', lambda c, o: (cache, Prepared(), '1'*64))
    monkeypatch.setattr(local_runtime_storage, 'require_training_storage', lambda p: None)
    monkeypatch.setattr(local_checkpoint, 'check_paths', lambda p: None)
    report = module.run(dict(input_manifest_sha256='manifest'), tmp_path)
    assert report['recordings'] == 513
    assert json.loads((tmp_path / 'progress.json').read_text())['completed'] == 512
    assert json.loads((tmp_path / 'MANIFEST.json').read_text()) == report
    assert len((tmp_path / 'inputs.jsonl').read_text().splitlines()) == 513
