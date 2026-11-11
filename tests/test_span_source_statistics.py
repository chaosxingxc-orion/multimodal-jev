from copy import deepcopy
import pytest
from multimodal_jev.span_source_statistics import paired_source_interval


def fixture():
    rows=[];before=[];after=[]
    # Two correlated rows per source. One source improves, one degrades.
    for source in range(2):
        for k in range(2):
            uid=f'{source}:{k}'
            rows.append(dict(id=uid,arm='structured',targets=[True],supervised=[True],provenance=dict(source_group=str(source))))
            before.append(dict(id=uid,predictions=[bool(source)]))
            after.append(dict(id=uid,predictions=[not bool(source)]))
    return rows,before,after


def test_source_clusters_preserved_and_perfect_pair_has_zero_uncertainty():
    rows,a,b=fixture();r=paired_source_interval(rows,a,b)
    assert r['arms']['structured']['request_weighted_delta_ci95']==[-1.,1.]
    assert r['arms']['structured']['request_weighted_delta']==0
    identical=paired_source_interval(rows,a,a)
    assert identical['arms']['structured']['request_weighted_delta_ci95']==[0.,0.]
    assert r==paired_source_interval(rows,a,b)


def test_repeating_rows_inside_each_source_does_not_invent_more_information():
    rows,a,b=fixture();expected=paired_source_interval(rows,a,b)['arms']['structured']
    for collection in (rows,a,b):
        repeated=deepcopy(collection)
        for r in repeated:r['id']+=':repeat'
        collection.extend(repeated)
    actual=paired_source_interval(rows,a,b)['arms']['structured']
    assert actual['requests']==2*expected['requests']
    for key in ('source_groups','request_weighted_delta_ci95','source_macro_delta_ci95'):
        assert actual[key]==expected[key]


@pytest.mark.parametrize('fault',['missing','duplicate','not_boolean','one_source','unreviewed'])
def test_bad_pair_rejected(fault):
    rows,a,b=fixture()
    if fault=='missing':b.pop()
    elif fault=='duplicate':b.append(b[0])
    elif fault=='not_boolean':b[0]['predictions']=[1]
    elif fault=='one_source':
        for row in rows:row['provenance']['source_group']='one'
    else:rows[0]['supervised']=[False]
    with pytest.raises(ValueError):paired_source_interval(rows,a,b)
