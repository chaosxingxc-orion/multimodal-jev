from copy import deepcopy
import math
from pathlib import Path
import sys

import pytest

from multimodal_jev.fsc_independent_audit import score, transitions, exact_p, require_same_transitions, VIEWS
from multimodal_jev.six_view_score import score_six_views
from test_six_view_score import fixture


def test_independent_raw_score_matches_original_and_ignores_forged_value():
    rows,targets,predictions=fixture()
    for p in predictions:p['value']='forged cached value'
    predictions[3]['raw_text']='```json\n{"value":null}\n```'
    predictions[4]['raw_text']='{"value":1}'
    actual=score(rows,targets,predictions)
    expected=score_six_views(rows,targets,predictions)
    assert actual['metrics']==expected['replicates'][0]
    assert actual['source_groups']==1 and actual['requests']==6
    assert actual['metrics']['by_variant']['missing_candidate']['correct']==0
    assert actual['metrics']['by_variant']['noul.True']['schema_valid']==0


@pytest.mark.parametrize('raw', ['bad','null','[]','{"value":null,"value":null}',
    '{"value":null,"extra":0}','{"value":null,"confidence":true}',
    '{"value":null,"confidence":-1}','{"value":null,"confidence":1e999}',
    '{"value":NaN}','{"value":1e999}'])
def test_invalid_json_not_counted_as_valid_null(raw):
    rows,targets,predictions=fixture();predictions[3]['raw_text']=raw
    report=score(rows,targets,predictions)
    assert report['metrics']['by_variant']['missing_candidate']['format_valid']==0
    assert report['metrics']==score_six_views(rows,targets,predictions)['replicates'][0]


@pytest.mark.parametrize('kind',['missing','duplicate','extra_target','binding','bool_rep','flags','view','audio'])
def test_full_bound_population_and_flags_required(kind):
    rows,targets,predictions=fixture()
    if kind=='missing':predictions.pop()
    if kind=='duplicate':predictions.append(deepcopy(predictions[0]))
    if kind=='extra_target':
        target=deepcopy(targets[0]);target['id']='extra';targets.append(target)
    if kind=='binding':predictions[0]['request_sha256']='wrong'
    if kind=='bool_rep':predictions[0]['replicate']=False
    if kind=='flags':predictions[0]['format_valid']=False
    if kind=='view':targets[0]['metadata']['variant']='rebind'
    if kind=='audio':targets[0]['metadata']['audio_sha256']='different'
    with pytest.raises(ValueError):score(rows,targets,predictions)


def test_recording_pairing_distinguishes_equal_marginal_accuracies():
    parent={str(i):dict.fromkeys(VIEWS,i<2) for i in range(4)}
    candidate={str(i):dict.fromkeys(VIEWS,i%2==0) for i in range(4)}
    result=transitions(parent,candidate)
    expected=dict(both_correct=1,both_wrong=1,fixed=1,regressed=1,mcnemar_exact_two_sided_p=1.)
    assert result['all_six_by_source_group']==expected
    assert all(cell==expected for cell in result['by_view'].values())
    assert sum(result['all_six_by_source_group'][f] for f in ('both_correct','both_wrong','fixed','regressed'))==4
    require_same_transitions(result,deepcopy(result))
    wrong=deepcopy(result);wrong['by_view']['native']['fixed']=2
    with pytest.raises(ValueError):require_same_transitions(result,wrong)
    wrong=deepcopy(result);wrong['by_view']['native']['mcnemar_exact_two_sided_p']=.5
    with pytest.raises(ValueError):require_same_transitions(result,wrong)


def test_all_six_uses_complete_recording_intersection():
    parent={'g':dict.fromkeys(VIEWS,True)};candidate=deepcopy(parent)
    candidate['g']['missing_candidate']=False
    result=transitions(parent,candidate)
    assert result['all_six_by_source_group']['regressed']==1
    assert result['by_view']['native']['both_correct']==1
    del candidate['g']['rebind']
    with pytest.raises(ValueError):transitions(parent,candidate)


@pytest.mark.parametrize('fixed,regressed,expected',[(0,0,1.),(4,0,.125),(0,4,.125),(3,1,.625),(3793,0,0.)])
def test_integer_exact_binomial_tail(fixed,regressed,expected):
    assert exact_p(fixed,regressed)==expected


def test_integer_tail_matches_frozen_float_algorithm_at_full_population():
    sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
    from select_direct_prompt import mcnemar_p
    for a,b in [(1900,1893),(1752,36),(50,30),(2561,1232)]:
        assert math.isclose(exact_p(a,b),mcnemar_p(a,b),rel_tol=1e-9,abs_tol=1e-12)


def test_cli_refuses_failed_or_incomplete_queue_before_observation(tmp_path,monkeypatch):
    sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
    import audit_fsc_test_comparison as cli
    monkeypatch.setattr(cli.subprocess,'check_output',lambda *_a,**_k:pytest.fail('must not inspect'))
    with pytest.raises(FileNotFoundError):cli.run(tmp_path)
    (tmp_path/'evaluation-terminal.json').write_text('{"exit_code":1}')
    with pytest.raises(ValueError,match='completed'):cli.run(tmp_path)
