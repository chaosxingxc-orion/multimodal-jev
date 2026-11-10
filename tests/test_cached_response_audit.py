import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

torch=pytest.importorskip('torch')
spec=importlib.util.spec_from_file_location('response_audit',Path(__file__).parents[1]/'tools/jev/audit_rollout_logprobs.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_cached_audit_uses_prefill_audio_once_and_correct_causal_positions():
    # Hand-written transition logits make off-by-one response alignment observable.
    matrix=torch.arange(49,dtype=torch.float32).reshape(7,7).sin()
    calls=[];cache=object()
    def model(**kw):
        calls.append(kw)
        return SimpleNamespace(logits=matrix[kw['input_ids']],past_key_values=cache)
    inputs={'input_ids':torch.tensor([[1,2,3,4,5]]),'attention_mask':torch.ones(1,5),
            'position_ids':torch.arange(5)[None], 'input_features':torch.zeros(1,2,4),
            'input_features_mask':torch.ones(1,4)}
    actual=module.cached_response_logprobs(model,inputs,2)
    expected=torch.stack([matrix[2].log_softmax(0)[3],matrix[3].log_softmax(0)[4],
                          matrix[4].log_softmax(0)[5]])[None]
    torch.testing.assert_close(actual,expected)
    assert len(calls)==3 and calls[0]['input_ids'].tolist()==[[1,2]]
    assert [x['position_ids'].tolist() for x in calls[1:]]==[[[2]],[[3]]]
    assert [x['attention_mask'].shape[1] for x in calls]==[2,3,4]
    assert all('input_features' not in x and x['past_key_values'] is cache for x in calls[1:])
