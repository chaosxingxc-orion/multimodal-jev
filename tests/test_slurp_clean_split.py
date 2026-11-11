import copy

import pytest

from multimodal_jev.slurp_clean_split import partition


def row(uid, text, filename, filler='amy'):
    return dict(slurp_id=uid, sentence=text, scenario='email', action='sendemail',
        tokens=[dict(id=0, surface=filler)], entities=[dict(type='person', span=[0])],
        recordings=[dict(file=filename)])


def req(filename, value):
    return dict(id='slurp:' + filename, split='train', audio_sha256=value * 64)


def test_transitive_closure_and_conflicts_and_deterministic_order():
    rows=[row(1,'Send MAIL','a'),row(2,' send  mail ','b'),
          row(3,'other text','c','bob'),row(4,'different','d')]
    requests=[req('a','a'),req('b','b'),req('c','b'),req('d','d')]
    result=partition(rows,requests)
    assert result==partition(list(reversed(rows)),list(reversed(requests)))
    indexed={r['id']:r for r in result['records']}
    assert len({indexed['slurp:'+f]['source_group'] for f in 'abc'})==1
    assert result['quarantine_ids']==['slurp:b','slurp:c']
    assert indexed['slurp:a']['supervision_status']=='available'
    assert indexed['slurp:d']['source_group']!=indexed['slurp:a']['source_group']


def test_duplicate_filename_does_not_select_arbitrary_label():
    result=partition([row(1,'one','a'),row(2,'two','a','bob')],[req('a','a')])
    assert len(result['records'])==1
    assert result['records'][0]['source_ids']==['1','2']
    assert result['records'][0]['supervision_status']=='conflict'


@pytest.mark.parametrize('bad',['split','missing','duplicate_source','duplicate_request','hash'])
def test_reject_incomplete_or_wrong_population(bad):
    rows=[row(1,'text','a')];requests=[req('a','a')]
    if bad=='split':requests[0]['split']='test'
    if bad=='missing':requests=[]
    if bad=='duplicate_source':rows+=copy.deepcopy(rows)
    if bad=='duplicate_request':requests+=copy.deepcopy(requests)
    if bad=='hash':requests[0]['audio_sha256']='bad'
    with pytest.raises(ValueError):partition(rows,requests)
