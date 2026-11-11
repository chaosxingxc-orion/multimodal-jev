"""The diagnostic must observe model execution without replacing its outputs."""
import pytest

torch = pytest.importorskip('torch')

from multimodal_jev.audio_stage_capture import capture_first


@pytest.mark.parametrize('before', [False, True])
def test_snapshot_is_first_call_and_does_not_alias_or_replace(tmp_path, before):
    module = torch.nn.Identity()
    handle = capture_first(module, 'stage', tmp_path, before=before)
    value = torch.tensor([[1., 2.]])
    result = module(value)
    assert result is value
    value.add_(10)
    module(torch.tensor([[99., 99.]]))
    saved = torch.load(tmp_path / 'stage.pt', weights_only=True)
    assert torch.equal(saved, torch.tensor([[1., 2.]]))
    handle.remove()


def test_tuple_output_keeps_entire_model_output(tmp_path):
    class Layer(torch.nn.Module):
        def forward(self, x):
            return (x + 1, 'metadata')
    module = Layer()
    handle = capture_first(module, 'stage', tmp_path)
    result = module(torch.tensor([3.]))
    assert result[1] == 'metadata'
    assert torch.equal(torch.load(tmp_path / 'stage.pt', weights_only=True), result[0])
    handle.remove()
