import copy
import importlib.util
from pathlib import Path
import pytest
pytest.importorskip('torch')
from test_simple_sampled_switch_audit import fixture

spec = importlib.util.spec_from_file_location('resident_batch_audit',
    Path(__file__).parents[1]/'tools/jev/verify_simple_resident_batch.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def inputs():
    outputs, _ = fixture()
    reference = outputs[1]
    result = copy.deepcopy(reference)
    result['records'] = result['records'][:128]
    result['reference_compared'] = False
    return result, reference, [r['request_id'] for r in result['records']]


def test_exact_scheduled_subset_and_full_probabilities():
    result, reference, ids = inputs()
    assert m.compare_rollout(result, reference, ids) == dict(requests=128, maximum_logprob_error=0.)


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'foreign', 'dtype', 'slot', 'probability', 'reference'])
def test_false_success_or_wrong_scheduled_outputs_rejected(fault):
    result, reference, ids = inputs()
    if fault == 'missing': result['records'].pop()
    elif fault == 'duplicate': result['records'][-1] = copy.deepcopy(result['records'][0])
    elif fault == 'foreign': result['records'][0]['request_identity'] = {'foreign': True}
    elif fault == 'dtype': result['worker'][0]['actual_lora_buffer_dtypes']['q'] = ['torch.float32']
    elif fault == 'slot': result['slot'] = 99
    elif fault == 'probability': result['records'][0]['candidate_logprobs'][0][0] += .1
    else: result['reference_compared'] = True
    with pytest.raises(ValueError): m.compare_rollout(result, reference, ids)
