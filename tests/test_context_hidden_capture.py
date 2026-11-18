import torch
import pytest

from multimodal_jev.context_decision_head import ContextDecisionHead
from multimodal_jev.context_pooling import ContextHistoryReadout
from multimodal_jev import context_hidden_capture as capture


def test_one_shot_capture_preserves_probabilities_and_rejects_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, 'require_training_storage', lambda p: None)
    def commit(path, writer):
        with path.open('xb') as stream:
            writer(stream)
    monkeypatch.setattr(capture, 'commit_file', commit)
    head = ContextDecisionHead(12, 3, 24).eval()
    r = ContextHistoryReadout(head, 99, 'a'*64)
    h = torch.randn(5, 12)
    p = torch.tensor([1, 2, 99, 3, 99])
    e = dict(policy_id='a'*64, marker_positions=[2, 4], expected_input_ids=p.tolist())
    expected = r(h, [5], [p], [e])[0]
    path = tmp_path/'hidden.pt'
    capture.capture_next_readout(r, path)
    with pytest.raises(ValueError): capture.capture_next_readout(r, tmp_path/'other.pt')
    actual = r(h, [5], [p], [e])[0]
    torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    v = torch.load(path, weights_only=True)
    assert torch.equal(v['hidden'], h) and v['extras'] == [e]
    assert torch.equal(v['probabilities'][0], expected)
    saved = path.read_bytes()
    r(h+1, [5], [p], [e])
    assert path.read_bytes() == saved and not r._jev_capture_pending
    with pytest.raises(ValueError): capture.capture_next_readout(r, path)
