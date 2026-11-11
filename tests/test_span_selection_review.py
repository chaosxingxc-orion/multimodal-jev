from copy import deepcopy
import pytest
from multimodal_jev.span_selection_review import select_review_packet


def fixture():
    requests=[];targets=[];metadata=[]
    for i in range(16):
        filename=f'{i}.flac';uid='slurp:'+filename
        requests.append(dict(id=uid,source_group=f'group{i}',audio_sha256=f'audio{i}',role='selection',official_split='train'))
        entities=[dict(type=k) for bit,k in enumerate(('date','time','timeofday')) if (i%8)&(1<<bit)]
        targets.append(dict(id=uid,annotation_status='available',optimization_allowed=False,
                            native_target=dict(entities=entities),source_ids=[str(i)]))
        metadata.append(dict(slurp_id=i,sentence=f'utterance {i}',recordings=[dict(file=filename)]))
    learning=[dict(source_group='training',audio_sha256='learning-audio',role='learning')]
    return requests,targets,metadata,learning


def test_deterministic_distinct_sources_and_no_automatic_review():
    args=fixture();rows,report=select_review_packet(*args,per_stratum=1)
    assert len(rows)==8 and len({r['request']['source_group'] for r in rows})==8
    assert all(set(r['review_status'].values())=={'pending'} for r in rows)
    assert not report['training_admitted'] and not report['evaluation_admitted']
    assert all(v['available_source_groups']==2 and v['selected']==1 for v in report['strata'].values())
    reversed_args=[list(reversed(x)) for x in args]
    assert select_review_packet(*reversed_args,per_stratum=1)==(rows,report)


@pytest.mark.parametrize('fault',['overlap_group','overlap_audio','wrong_role','missing_target','bad_reference'])
def test_cross_role_or_incomplete_provenance_rejected(fault):
    args=deepcopy(fixture());requests,targets,metadata,learning=args
    if fault=='overlap_group':requests[0]['source_group']='training'
    elif fault=='overlap_audio':requests[0]['audio_sha256']='learning-audio'
    elif fault=='wrong_role':requests[0]['role']='learning'
    elif fault=='missing_target':targets.pop()
    else:metadata[0]['recordings']=[dict(file='other.flac')]
    with pytest.raises(ValueError):select_review_packet(*args)


def test_rare_strata_report_shortfall_without_duplicate_upsampling():
    args=fixture();rows,report=select_review_packet(*args,per_stratum=3)
    assert len(rows)==16
    assert all(v['shortfall']==1 for v in report['strata'].values())
