import json
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).parents[1] / 'tools/jev'))
from analyze_omni_precision_logits import CELLS, FLAGS, analyze, digest


def fixture_run(tmp_path):
    identity = dict(keys=[[str(i), 'decision'] for i in range(6)],
                    cells=[[n, *f] for n, f in zip(CELLS, FLAGS)],
                    response_ids=[0, 1], targets_read=False, weight_updates=0,
                    preflight_only=False)
    path = tmp_path / 'identity.json'
    path.write_text(json.dumps(identity))
    return identity, digest(path)


def write_record(root, identity, row, cell, logits, previous=None):
    path = root / f'row-{row:02d}-{cell}-logits.pt'
    torch.save(logits, path)
    labels = torch.tensor(identity['response_ids']).unsqueeze(-1)
    high = logits.float().log_softmax(-1).gather(-1, labels).squeeze(-1)
    low = logits.log_softmax(-1).gather(-1, labels).squeeze(-1).float()
    delta = None if previous is None else float((high - previous.float().log_softmax(-1)
            .gather(-1, labels).squeeze(-1)).abs().max())
    record = dict(row=row, cell=cell, key=identity['keys'][row], seconds=1.,
                  logits_sha256=digest(path), logits_dtype=str(logits.dtype),
                  fp32_log_probs=high.tolist(), native_dtype_log_probs=low.tolist(),
                  normalization_max_error=float((high-low).abs().max()),
                  predecessor_max_error=delta)
    target = root / f'row-{row:02d}-{cell}.json'
    target.write_text(json.dumps(record))
    return target


def test_equal_response_probabilities_do_not_imply_equal_full_logits(tmp_path):
    identity, pin = fixture_run(tmp_path)
    logits = torch.tensor([[0., 2., 3., 4.], [2., 0., 3., 4.]], dtype=torch.bfloat16)
    changed = logits.clone(); changed[:, [2, 3]] = changed[:, [3, 2]]
    write_record(tmp_path, identity, 0, 'reference', logits)
    write_record(tmp_path, identity, 0, 'autocast', changed, logits)
    report = analyze(tmp_path, pin)
    assert report['status'] == 'partial_artifacts'
    assert report['parity_accepted'] is False
    last = report['records'][-1]
    assert last['predecessor_response_max_error'] == 0
    assert last['predecessor_full_logits_max_error'] == 1
    assert last['predecessor_logits_equal'] is False


@pytest.mark.parametrize('fault', ['hash', 'probability', 'identity', 'gap'])
def test_rejects_corrupt_or_incomplete_evidence(tmp_path, fault):
    identity, pin = fixture_run(tmp_path)
    logits = torch.tensor([[0., 2.], [2., 0.]], dtype=torch.bfloat16)
    path = write_record(tmp_path, identity, 0, 'reference', logits)
    record = json.loads(path.read_text())
    if fault == 'hash':
        record['logits_sha256'] = '0' * 64
    elif fault == 'probability':
        record['fp32_log_probs'][0] += .5
    elif fault == 'identity':
        record['key'] = ['wrong', 'decision']
    elif fault == 'gap':
        path.unlink()
        path = tmp_path / 'row-00-autocast.json'
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        analyze(tmp_path, pin)


def test_complete_artifacts_are_never_parity_acceptance(tmp_path):
    identity, pin = fixture_run(tmp_path)
    logits = torch.tensor([[0., -12.], [-12., 0.]], dtype=torch.bfloat16)
    for row in range(6):
        for cell in CELLS:
            write_record(tmp_path, identity, row, cell, logits,
                         None if cell == 'reference' else logits)
    report = analyze(tmp_path, pin)
    assert report['completed_cells'] == 30
    assert report['status'] == 'artifacts_complete'
    assert report['parity_accepted'] is False
    assert report['records'][0]['native_dtype_log_probs'] == [0., 0.]
    assert all(v < 0 for v in report['records'][0]['fp32_log_probs'])
