import math
import pytest
from multimodal_jev.final_probability_diagnostics import diagnose


def fixture():
    classes = [f'c{i}' for i in range(60)]
    rows = [dict(id=str(i), audio_sha256=str(i), intent=classes[i]) for i in range(2)]
    records = [dict(id=r['id'], audio_sha256=r['audio_sha256'], policy_id='p', view='direct',
        variant='native', status='completed', prediction='c0', candidate_values=classes,
        candidate_logprobs=[-math.log(60)]*60) for r in rows]
    return rows, classes, records


def test_uniform_distribution_analytic_metrics():
    rows, classes, records = fixture()
    result = diagnose(rows, classes, records, policy_id='p')
    assert result['nll_nats'] == pytest.approx(math.log(60))
    assert result['multiclass_brier_sum_over60'] == pytest.approx(59/60)
    assert result['top_label_ece'] == pytest.approx(.5-1/60)
    assert result['incorrect_mean_confidence'] == pytest.approx(1/60)
    assert result['predicted_classes'] == 1 and result['zero_recall_supported_classes'] == 1
    assert sum(b['count'] for b in result['reliability_bins']) == 2


def test_confident_wrong_worsens_quality_despite_same_argmax_accuracy():
    rows, classes, records = fixture()
    baseline = diagnose(rows, classes, records, policy_id='p')
    for r in records: r['candidate_logprobs'] = [math.log(.99)]+[math.log(.01/59)]*59
    result = diagnose(rows, classes, records, policy_id='p')
    assert result['accuracy'] == baseline['accuracy']
    assert result['nll_nats'] > baseline['nll_nats']
    assert result['incorrect_confidence_ge_0_9'] == 1
    assert result['incorrect_mean_confidence'] == pytest.approx(.99)


@pytest.mark.parametrize('fault', ['missing', 'no_probabilities', 'failed', 'normalization'])
def test_no_partial_probability_report(fault):
    rows, classes, records = fixture()
    if fault == 'missing': records.pop()
    if fault == 'no_probabilities': records[0].pop('candidate_logprobs')
    if fault == 'failed': records[0].update(status='failed', error='failure')
    if fault == 'normalization': records[0]['candidate_logprobs'] = [-1.]*60
    with pytest.raises(ValueError): diagnose(rows, classes, records, policy_id='p')
