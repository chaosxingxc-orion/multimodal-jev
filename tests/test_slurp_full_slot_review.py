import hashlib
from pathlib import Path
import runpy

import pytest

m=runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/jev/bind_slurp_full_slot_review.py'))


def fixture():
    source=dict(source_id='1',metadata_sha256='metadata',reference='turn off socket')
    record=dict(source_id='1',metadata_sha256='metadata',reference_sha256=hashlib.sha256(source['reference'].encode()).hexdigest(),
        optimization_allowed=False,actor_input_allowed=False,independent_semantic_review=False,modality='reference_text',
        positive={'device':dict(quote='socket',rationale='Explicit hardware')},negative_types=['place'],
        unavailable={'event':dict(quote='turn off',rationale='Activity scope unclear',reason_kind='definition')},
        negative_coverage=dict(explicitly_assessed_all55=True,basis='Reviewed every supplied definition',source_summary=source['reference']))
    return source,record


def test_explicit_partition_validated_without_inventing_negatives():
    source,record=fixture()
    m['validate_record'](record,source,{'device','event','place'})
    record['negative_types']=[]
    with pytest.raises(ValueError,match='complete type'):m['validate_record'](record,source,{'device','event','place'})


@pytest.mark.parametrize('change',['overlap','duplicate','bad_quote','stale','admitted','independent','coverage'])
def test_invalid_evidence_and_admission_rejected(change):
    source,record=fixture()
    if change=='overlap':record['negative_types'].append('event')
    elif change=='duplicate':record['negative_types']*=2
    elif change=='bad_quote':record['positive']['device']['quote']='not spoken'
    elif change=='stale':record['metadata_sha256']='stale'
    elif change=='admitted':record['optimization_allowed']=True
    elif change=='independent':record['independent_semantic_review']=True
    else:record['negative_coverage']['explicitly_assessed_all55']=False
    with pytest.raises(ValueError):m['validate_record'](record,source,{'device','event','place'})
