import copy

import pytest

from tools.reports.prepare_slurp_context_test_inputs import select_inputs


def fixture():
    native=[dict(id='slurp:'+str(i)+'.flac',input=dict(audio='/audio/'+str(i)+'.flac',schema_sha256='schema'),
                 audio_sha256=str(i),source_group='source'+str(i),split='test') for i in range(3)]
    population=dict(ids=[native[1]['id'],native[0]['id']],excluded_ids=[native[2]['id']])
    return native,population,[dict(id='learn',audio_sha256='learn')],[dict(id='select',audio_sha256='select')]


def test_exclusion_only_preserves_predeclared_order_and_no_overlap():
    args=fixture();selected,overlaps=select_inputs(*args)
    assert [r['id'] for r in selected]==args[1]['ids']
    assert all(not any(v.values()) for v in overlaps.values())


@pytest.mark.parametrize('fault',['missing','extra','duplicate','kept_excluded','bad_split','target_field',
                                'input_label','noncanonical','learning_id','selection_hash','duplicate_source'])
def test_identity_or_leakage_rejected(fault):
    args=list(copy.deepcopy(fixture()));row=args[0][0]
    if fault=='missing':args[1]['ids'].pop()
    elif fault=='extra':args[1]['ids'].append('unknown')
    elif fault=='duplicate':args[0].append(row)
    elif fault=='kept_excluded':args[1]['excluded_ids'].append(row['id'])
    elif fault=='bad_split':row['split']='train'
    elif fault=='target_field':row['target']='gold'
    elif fault=='input_label':row['input']['intent']='gold'
    elif fault=='noncanonical':row['input']['audio']='/audio/changed.flac'
    elif fault=='learning_id':args[2][0]['id']=row['id']
    elif fault=='selection_hash':args[3][0]['audio_sha256']=row['audio_sha256']
    else:args[2].append(args[2][0])
    with pytest.raises(ValueError):select_inputs(*args)
