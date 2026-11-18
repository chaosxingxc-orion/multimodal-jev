import copy
import json
from pathlib import Path

import pytest
import torch

from multimodal_jev.context_reference_dataset import digest
from tools.reports import context_replica_transfer_client as m
from test_context_macro_evaluation import setup
from test_context_replica_macro_rollout import pool
from test_context_transfer_macro_evaluation import source_config, write


def fixture(tmp_path, monkeypatch, dataset):
    c = source_config(tmp_path)
    _, population, encode, args, clients = setup(tmp_path, monkeypatch, n=129)
    engine = pool(tmp_path/'replica-journal', monkeypatch, clients)
    width = m.SIZES[dataset][1]
    candidates = [f'c{i}' for i in range(width)]
    def expanded(asset):
        row, receipt = encode(asset)
        row['positions'] = torch.arange(width).reshape(1, width)
        receipt.update(candidate_ids=candidates, positions=list(range(width)))
        return row, receipt
    receipts = [expanded(a)[1] for a in population]
    for client in clients:
        original = client.score_many
        def score(batch, _original=original, **kwargs):
            records = _original(batch, **kwargs)
            for record in records:
                record.update(prob=[.5]*width, logits=[0.]*width)
            return records
        client.score_many = score
    admitted = dict(dataset=dataset, population=population, input_receipts=receipts,
        candidate_ids=candidates, processor_sha256='fixture', cache_root=str(tmp_path/'cache'))
    indices = [0, 31, 128]
    selected = dict(dataset=dataset, assets=[population[i] for i in indices],
        input_receipts=[receipts[i] for i in indices], candidate_ids=candidates)
    reference = dict(schema='jev-target-width-hf-reference-v1', policy_id='p0',
        panel_sha256=digest(selected), records=[dict(id=population[i]['id'],
            request_identity=dict(policy_id='p0', positions=receipts[i]['positions'], inputs=receipts[i]['inputs']),
            prob=[.5]*width, logits=[0.]*width) for i in indices])
    bundle = dict(policy_id='p0', contract={})
    monkeypatch.setattr(m, 'validate_config', lambda _: bundle)
    monkeypatch.setattr(m, 'inputs', lambda _: (admitted, selected, {'fixture': 'coverage'}))
    monkeypatch.setattr(m, 'encoder', lambda *a: expanded)
    monkeypatch.setattr(m, 'check_paths', lambda _: None)
    monkeypatch.setattr(m, 'connect_pool', lambda *a, **k: engine)
    monkeypatch.setattr(m, 'hf_reference', lambda *a: reference)
    source = json.loads(Path(c['source_evaluation']['path']).read_text())
    c.update(dataset=dataset, source_release={'sha256': source['experiment']['release_sha256']},
        output=str(tmp_path/'hf'), portable_full_inputs=dict(manifest_sha256='m', population_evidence_sha256='e'),
        dual_runtime_identity=args['runtime_identity'], macro_client={'runtime_identity': args['runtime_identity']},
        audio_token_id=9, macro_queries=128)
    m.run(c, {}, 'hf')
    c.update(output=str(tmp_path/'macro'), hf_reference=dict(path=str(tmp_path/'hf/reference.json'),
        sha256=m.file_hash(tmp_path/'hf/reference.json')))
    return c, engine, clients, selected, reference


@pytest.mark.parametrize('dataset', list(m.SIZES))
def test_real_replica_macro128_rollout32_and_unpadded_tail(tmp_path, monkeypatch, dataset):
    c, engine, clients, _, _ = fixture(tmp_path, monkeypatch, dataset)
    result = m.run(c, {}, 'macro')
    assert result['recordings'] == 129 and result['decisions'] == m.SIZES[dataset][1]
    assert result['target_scored'] is False
    assert sorted(len(call) for client in clients for call in client.calls) == [1, 32, 32, 32, 32]
    assert all(client.phase == 'closed' for client in clients)
    audit = m.audit_outputs(c, tmp_path/'macro', engine.root, tmp_path/'hf')
    assert audit['inference_audit']['batches'] == 2 and audit['parity']['exact']
    assert audit['runtime_provenance_verified'] is False and audit['target_scored'] is False


@pytest.mark.parametrize('fault', ['source', 'hf-policy', 'hf-panel', 'logits', 'tail'])
def test_failed_source_parity_or_tail_never_publishes_success(tmp_path, monkeypatch, fault):
    c, engine, clients, _, ref = fixture(tmp_path, monkeypatch, 'FSC_native31')
    if fault == 'source':
        c['policy_id'] = 'wrong'
        monkeypatch.setattr(m, 'inputs', lambda _: pytest.fail('target accessed before source admission'))
    elif fault in ('hf-policy', 'hf-panel', 'logits'):
        ref = copy.deepcopy(ref)
        if fault == 'hf-policy': ref['policy_id'] = 'wrong'
        elif fault == 'hf-panel': ref['panel_sha256'] = 'wrong'
        else: ref['records'][0]['logits'][0] = 1e-9
        c['hf_reference'] = write(tmp_path/'hf/reference.json', ref)
    else:
        original = m.encoder
        def failed_encoder(*a):
            encode = original(*a)
            def fail_tail(asset):
                if asset['id'] == '128': raise ValueError('missing tail cache')
                return encode(asset)
            return fail_tail
        monkeypatch.setattr(m, 'encoder', failed_encoder)
    with pytest.raises(ValueError): m.run(c, {}, 'macro')
    assert not (tmp_path/'macro/result.json').exists()
    if fault != 'source':
        assert m.read(tmp_path/'macro/terminal.json')['exit_code'] == 1
    if fault in ('source', 'hf-policy', 'hf-panel'):
        assert not any(client.calls for client in clients)
    else:
        assert all(client.phase == 'closed' for client in clients)


@pytest.mark.parametrize('fault', ['evidence', 'panel', 'source', 'parity', 'prediction', 'result'])
def test_independent_collector_detects_tampering(tmp_path, monkeypatch, fault):
    c, engine, _, _, _ = fixture(tmp_path, monkeypatch, 'MINDS_native14')
    m.run(c, {}, 'macro')
    paths = dict(evidence='input-evidence.json', panel='panel.json', source='source-binding.json',
        parity='parity.json', prediction='inference/predictions.json', result='result.json')
    p = tmp_path/'macro'/paths[fault]
    value = m.read(p)
    if fault == 'evidence': value['inputs']['manifest_sha256'] = 'wrong'
    elif fault == 'panel': value['assets'].pop()
    elif fault == 'source': value['role'] = 'best_target'
    elif fault == 'parity': value['exact'] = False
    elif fault == 'prediction': value['records'].pop()
    else: value['target_scored'] = True
    write(p, value)
    with pytest.raises(ValueError): m.audit_outputs(c, tmp_path/'macro', engine.root, tmp_path/'hf')


@pytest.mark.parametrize('fault', ['complex', 'schema', 'runtime', 'macro', 'rollout', 'replicas', 'policy'])
def test_unsupported_protocol_refused_before_policy_load(monkeypatch, fault):
    c = dict(dataset='FSC_native31', macro_queries=128, source_policy='s', snapshot='s',
        source_policy_id='p', policy_id='p', source_contract={}, contract={}, dual_runtime_identity={},
        macro_client=dict(schema='jev-replica-macro-client-v1', max_queries=128,
            rollout_queries=32, replicas=[{}, {}], runtime_identity={}))
    if fault == 'complex': c['dataset'] = 'Audio2Tool'
    elif fault == 'schema': c['macro_client']['schema'] = 'jev-macro-client-v1'
    elif fault == 'runtime': c['macro_client']['runtime_identity'] = {'other': True}
    elif fault == 'macro': c['macro_queries'] = 32
    elif fault == 'rollout': c['macro_client']['rollout_queries'] = 4
    elif fault == 'replicas': c['macro_client']['replicas'].pop()
    else: c['source_policy_id'] = 'wrong'
    monkeypatch.setattr(m, 'read_execution_bundle', lambda *a, **k: pytest.fail('unexpected policy access'))
    with pytest.raises(ValueError): m.validate_config(c)


@pytest.mark.parametrize('fault', [None, 'missing', 'changed', 'receipt'])
def test_readonly_real_cache_no_waveform_or_reprocessing(tmp_path, monkeypatch, fault):
    from test_cached_context_preparation import setup as cached_setup
    from multimodal_jev.encoded_audio_cache import key_for
    from multimodal_jev.slurp_training import content_hash
    from multimodal_jev.span_request_identity import tensor_identity
    from multimodal_jev import cached_context_preparation
    request, audio_hash, factory, calls, audio = cached_setup(tmp_path, monkeypatch)
    prepare = factory(limit=4096)
    value = prepare((request, audio_hash))
    key = key_for(prepare.identity(request, audio_hash))
    path = prepare.cache.root/(key+'.pt')
    asset = dict(id='a', audio_sha256=audio_hash, request_sha256=content_hash(request.model_input()))
    original = dict(asset, cache_key=key, cache_sha256=m.file_hash(path),
        tensors={k: tensor_identity(v) for k,v in dict(value.inputs, positions=value.positions).items()})
    receipt = dict(asset, candidate_ids=['q'], positions=[2],
        inputs={k:v for k,v in original['tensors'].items() if k!='positions'})
    admitted = dict(cache_root=str(prepare.cache.root), requests={'a':request}, candidate_ids=['q'],
        cache_receipts=[original], input_receipts=[receipt], processor_sha256='a'*64)
    monkeypatch.setattr(cached_context_preparation, 'processor_cache_identity', lambda _: 'a'*64)
    audio.unlink()
    if fault == 'missing': path.unlink()
    elif fault == 'changed': path.write_bytes(b'changed')
    elif fault == 'receipt': receipt['positions']=[1]
    before = sorted(p.name for p in prepare.cache.root.iterdir())
    encode = m.encoder(dict(snapshot='fixture', audio_token_id=98), admitted, {'contract':{'token':{'id':99}}})
    if fault:
        with pytest.raises((ValueError, FileNotFoundError)): encode(asset)
    else:
        row, observed = encode(asset)
        assert observed == receipt and row['inputs']['input_features'].dtype == torch.bfloat16
    assert len(calls) == 1 and before == sorted(p.name for p in prepare.cache.root.iterdir())


@pytest.mark.parametrize('fault', ['dataset', 'release', 'cache'])
def test_incompatible_portable_binding_refused(monkeypatch, fault):
    full = dict(dataset='SLURP_clean_TEST60', source_release={'sha256':'v1'}, cache_root='/cache')
    panel = dict(cache_root='/cache')
    c = dict(dataset='SLURP_clean_TEST60', source_release={'sha256':'v1'},
        portable_full_inputs={}, portable_parity_inputs={})
    if fault == 'dataset': full['dataset']='FSC_native31'
    elif fault == 'release': full['source_release']={'sha256':'v2'}
    else: panel['cache_root']='/other-cache'
    monkeypatch.setattr(m, 'load_full', lambda _: full)
    monkeypatch.setattr(m, 'load_panel', lambda _: panel)
    with pytest.raises(ValueError): m.inputs(c)
