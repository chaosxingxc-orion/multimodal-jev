from pathlib import Path
import runpy

import pytest

mod=runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/jev/audit_slurp_question_information.py'))
analyze=mod['analyze'];fold=mod['fold']


def rows():
    return [dict(id=str(i),source_group=f'g{i}',intent='a' if i%2 else 'b',
                 types={'s'} if i%2 else set()) for i in range(20)]


def test_perfect_slot_signature_and_redundant_intent():
    data=rows()
    for result in (analyze(data,slots=['s']),analyze(data,intents=['a','b'])):
        assert result['conditional_entropy_bits']==0
        assert result['empirical_signature_majority_upper_bound']==1
        assert result['source_group_5fold_lookup']['accuracy']==1


def test_uninformative_slots_do_not_manufacture_information():
    result=analyze(rows(),slots=['absent'])
    assert result['conditional_entropy_bits']==1
    assert result['empirical_mutual_information_bits']==0
    assert result['empirical_signature_majority_upper_bound']==.5
    assert result['mixed_signature_recordings']==20


def test_source_replicates_cannot_leak_into_other_folds():
    groups=[f'group{i}' for i in range(30)]
    other=next(g for g in groups if fold(g)!=fold(groups[0]))
    data=[dict(id=str(i),source_group=groups[0],intent='a',types={'unique'}) for i in range(10)]
    data.append(dict(id='other',source_group=other,intent='b',types=set()))
    result=analyze(data,slots=['unique'])
    assert result['empirical_signature_majority_upper_bound']==1
    assert result['source_group_5fold_lookup']['accuracy']==0
    assert result['source_group_5fold_lookup']['unseen_signature_recordings']==11


def test_no_questions_has_one_signature_and_duplicates_rejected():
    assert analyze(rows())['observed_signatures']==1
    with pytest.raises(ValueError):analyze(rows(),slots=['s','s'])
    with pytest.raises(ValueError):analyze([rows()[0],rows()[0]])
