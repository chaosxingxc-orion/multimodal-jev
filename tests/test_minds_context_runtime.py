import copy
import json
from types import SimpleNamespace

import pytest
import torch

from multimodal_jev.slurp_training import content_hash
from tools.reports import minds_context_runtime as module


def fixture(count=300):
    rows = [dict(id=f'audio-{i}', source_group=f'group-{i//3}') for i in range(count)]
    receipts = [dict(id=r['id'], tokens=100+i, positions=[list(range(14))],
        tensors=dict(input_ids=dict(sha256=str(i)), positions=dict(sha256='marker')))
        for i, r in enumerate(rows)]
    return rows, receipts


def prediction(row, receipt):
    identity = dict(policy_id='policy', positions=receipt['positions'][0],
        inputs={k:v for k,v in receipt['tensors'].items() if k != 'positions'})
    return dict(id=row['id'], policy_id='policy', request_identity=identity,
        request_id=content_hash(identity), prob=[.2]*14)


def test_panel_uses_lengths_and_input_identity_without_labels():
    rows, receipts = fixture()
    indices = module.panel_indices(rows, receipts)
    assert len(indices) == len(set(indices)) == 256
    assert set(range(16)) | set(range(284, 300)) <= set(indices)
    permutation = list(reversed(range(len(rows))))
    permuted = module.panel_indices([rows[i] for i in permutation], [receipts[i] for i in permutation])
    assert {rows[i]['id'] for i in indices} == {rows[permutation[i]]['id'] for i in permuted}


@pytest.mark.parametrize('fault', ['nan', 'range', 'count', 'bool', 'policy', 'input', 'digest'])
def test_rejects_invalid_engine_evidence(fault):
    rows, receipts = fixture(1)
    p = prediction(rows[0], receipts[0])
    if fault == 'nan': p['prob'][0] = float('nan')
    elif fault == 'range': p['prob'][0] = 1.01
    elif fault == 'count': p['prob'].pop()
    elif fault == 'bool': p['prob'][0] = True
    elif fault == 'policy': p['policy_id'] = 'other'
    elif fault == 'input': p['request_identity']['inputs'] = {}
    else: p['request_id'] = 'other'
    with pytest.raises(ValueError):
        module.checked_prediction(rows[0], receipts[0], p, 'policy')


@pytest.mark.parametrize('fault', [None, 'short', 'policy', 'input', 'panel', 'delta'])
def test_admission_rechecks_actual_predeclared_panel(tmp_path, fault):
    rows, receipts = fixture()
    indices = module.panel_indices(rows, receipts)
    hf = dict(policy_id='policy', indices=indices, input_manifest_sha256='input',
        records=[prediction(rows[i], receipts[i]) for i in indices])
    vl = copy.deepcopy(hf)
    if fault == 'short': vl['records'].pop()
    elif fault == 'policy': vl['policy_id'] = 'other'
    elif fault == 'input': vl['records'][0]['request_identity']['inputs'] = {}
    elif fault == 'panel': hf['indices'] = hf['indices'][::-1]; vl['indices'] = hf['indices']
    elif fault == 'delta': vl['records'][0]['prob'][0] += 1e-7
    (tmp_path/'hf.json').write_text(json.dumps(hf))
    (tmp_path/'vllm.json').write_text(json.dumps(vl))
    config = dict(policy_id='policy', input_manifest_sha256='input')
    if fault:
        with pytest.raises(ValueError): module.require_parity(config, tmp_path, rows, receipts)
    else:
        assert module.require_parity(config, tmp_path, rows, receipts)['exact']


@pytest.mark.parametrize('failure', [False, True])
def test_inference_durable_records_progress_and_no_false_completion(tmp_path, monkeypatch, failure):
    from multimodal_jev import local_checkpoint
    monkeypatch.setattr(local_checkpoint, 'check_paths', lambda p: None)
    rows, receipts = fixture(129)
    class Engine:
        count = 0
        def score(self, inputs, positions, audio_token_id):
            i = self.count
            self.count += 1
            if failure and i == 128:
                raise ValueError('synthetic terminal failure')
            return prediction(rows[i], receipts[i])
    def encode(pair):
        row, receipt = pair
        return row, receipt, SimpleNamespace(inputs={}, positions=None)
    config = dict(policy_id='policy', source_role='initial', input_manifest_sha256='input')
    if failure:
        with pytest.raises(ValueError, match='synthetic terminal'):
            module.infer(config, tmp_path, rows, receipts, encode, Engine())
        assert not (tmp_path/'result.json').exists()
        assert not (tmp_path/'predictions.json').exists()
        assert len(list(tmp_path.glob('record-*.json'))) == 128
    else:
        result = module.infer(config, tmp_path, rows, receipts, encode, Engine())
        assert result['recordings'] == 129
        assert len(json.loads((tmp_path/'predictions.json').read_text())['records']) == 129
    assert json.loads((tmp_path/'progress.json').read_text())['completed'] == 128
