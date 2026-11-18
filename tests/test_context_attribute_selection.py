import copy
import pytest
from tools.reports.build_context_attribute_selection import validate_split


def case():
    learning=[dict(id='train',source_group='g1',audio_sha256='h1')]
    selection=[dict(id='selection',source_group='g2',audio_sha256='h2',role='selection',
                    official_split='train',known_pending_source=False,intent='a',scenario='s',action='x')]
    audio=[dict(id='selection',audio_sha256='h2')]
    return learning,selection,audio,dict(events=[dict(id='a')]),dict(a=['s','x'])


def test_disjoint_held_out_population():
    assert validate_split(*case(),expected_count=1)==1


@pytest.mark.parametrize('field',['id','source_group','audio_sha256'])
def test_cross_split_leakage_rejected(field):
    args=copy.deepcopy(case());args[1][0][field]=args[0][0][field]
    if field=='id':args[2][0]['id']=args[0][0][field]
    if field=='audio_sha256':args[2][0][field]=args[0][0][field]
    with pytest.raises(ValueError,match='overlaps TRAIN'):validate_split(*args,expected_count=1)


@pytest.mark.parametrize('field,value',[('role','learning'),('official_split','test'),
    ('known_pending_source',True),('action','wrong'),('intent','unknown')])
def test_bad_provenance_rejected(field,value):
    args=copy.deepcopy(case());args[1][0][field]=value
    with pytest.raises(ValueError):validate_split(*args,expected_count=1)


def test_missing_audio_rejected():
    args=list(case());args[2]=[]
    with pytest.raises(ValueError):validate_split(*args,expected_count=1)
