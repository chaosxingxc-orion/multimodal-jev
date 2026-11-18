import math
import pytest
from multimodal_jev.simple_final_metrics import score


def fixture():
    classes = [f'c{i}' for i in range(60)]
    rows = [dict(id=str(i), audio_sha256=f'h{i}', intent=classes[i]) for i in range(3)]
    records = [dict(id=r['id'], audio_sha256=r['audio_sha256'], policy_id='fixed',
        view='direct', variant='native', status='completed', prediction=r['intent']) for r in rows]
    return rows, classes, records


def test_failures_keep_denominator_and_fixed_absent_classes():
    rows, classes, records = fixture()
    records[1]['prediction'] = None
    records[2].update(status='failed', error='backend terminal failure')
    r = score(rows, classes, records, policy_id='fixed')
    assert r['accuracy'] == 1/3 and r['invalid'] == 2 and r['failed'] == 1
    assert r['macro_F1_fixed60'] == 1/60
    assert r['macro_recall_supported'] == 1/3
    assert r['per_class']['c59']['recall'] is None


@pytest.mark.parametrize('fault', ['missing','duplicate','policy','audio','pending','empty_error'])
def test_population_and_terminal_identity_rejection(fault):
    rows, classes, records = fixture()
    if fault == 'missing': records.pop()
    if fault == 'duplicate': records.append(records[0])
    if fault == 'policy': records[0]['policy_id']='other'
    if fault == 'audio': records[0]['audio_sha256']='other'
    if fault == 'pending': records[0]['status']='running'
    if fault == 'empty_error': records[0]['status']='failed'
    with pytest.raises(ValueError): score(rows, classes, records, policy_id='fixed')


def test_full_candidate_distribution_and_argmax_binding():
    rows, classes, records = fixture()
    records[0].update(candidate_values=classes, candidate_logprobs=[-math.log(60)]*60)
    assert score(rows, classes, records, policy_id='fixed')['accuracy'] == 1
    records[0]['prediction']='c1'
    with pytest.raises(ValueError, match='argmax'):
        score(rows, classes, records, policy_id='fixed')
    records[0]['candidate_values']=list(reversed(classes))
    with pytest.raises(ValueError, match='ordered'):
        score(rows, classes, records, policy_id='fixed')
