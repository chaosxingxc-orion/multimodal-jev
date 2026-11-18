import pytest

from tools.reports.audit_slurp_omni_cluster_reference import cluster_interval


def fixture():
    return [dict(id=str(i), source_group='large', correct=True) for i in range(9)] + [
        dict(id='9', source_group='small', correct=False)]


def test_unequal_clusters_keep_recording_weights():
    result = cluster_interval(fixture())
    assert result['accuracy'] == .9
    assert result['source_groups'] == 2
    assert result['percentile95'] == [0., 1.]


def test_order_invariant():
    assert cluster_interval(fixture()) == cluster_interval(list(reversed(fixture())))


@pytest.mark.parametrize('kind', ['duplicate', 'missing_group', 'non_boolean', 'one_group'])
def test_bad_population(kind):
    rows = fixture()
    if kind == 'duplicate':rows.append(rows[0])
    elif kind == 'missing_group':rows[0]['source_group'] = ''
    elif kind == 'non_boolean':rows[0]['correct'] = 1
    else:
        for row in rows:row['source_group'] = 'same'
    with pytest.raises(ValueError):cluster_interval(rows)
