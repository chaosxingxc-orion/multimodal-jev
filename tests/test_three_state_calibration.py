from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from build_three_state_calibration import reference_coordinates, select
from score_three_state_calibration import summarize
import json
from consolidate_intent_presence_references import sha


def test_balanced_selection_is_order_independent_and_distinct_by_source():
    coordinates=[dict(family=f,source_id=s+str(i),semantic_id=q,expected_status=s)
        for f in ('I_present','S_present') for s in ('positive','negative','unavailable')
        for i in range(5) for q in ('a','b')]
    chosen=select(coordinates)
    assert chosen==select(list(reversed(coordinates))) and len(chosen)==12
    for f in ('I_present','S_present'):
        for s in ('positive','negative','unavailable'):
            assert len({c['source_id'] for c in chosen if c['family']==f and c['expected_status']==s})==2
    with pytest.raises(ValueError,match='insufficient'):
        select([c for c in coordinates if c['expected_status']!='unavailable'])


def test_stale_conflicting_and_missing_reference_are_not_promoted():
    def intent(status,incompatible=False):
        return dict(incompatible_fields=['card'] if incompatible else [],optimization_allowed=False,
            decision=dict(status=status,source_id='a',semantic_id='i',reference='text',optimization_allowed=False))
    def slot(value,applicability='current_semantics_reference_only'):
        return dict(applicability=applicability,optimization_allowed=False,proposed_value=value,
            source_id='b',definition_id='s',original_assessment=dict(reference='text'))
    assert reference_coordinates([intent('negative'),intent('positive')],[])==[]
    result=reference_coordinates([intent('positive',True)],[slot(None),slot(False,'stale_semantics_requires_rereview')])
    assert len(result)==1 and result[0]['expected_status']=='unavailable'
    result=reference_coordinates([intent('negative'),intent('negative')],[slot(True)])
    assert len(result)==2 and len(result[0]['assessment_sha256'])==2
    with pytest.raises(ValueError,match='invalid reference value'):
        reference_coordinates([],[slot(0)])


def test_paired_score_keeps_missing_and_separates_absence_from_ambiguity():
    cases=[dict(request_id=str(i),semantic_id='q',family='S_present',expected_status=s)
           for i,s in enumerate(('positive','negative','unavailable'))]
    answers={'a':{'0':{'q':'positive'},'1':{'q':'unavailable'}},
             'b':{'0':{'q':'negative'},'1':{'q':'negative'},'2':{'q':'unavailable'}}}
    result=summarize(cases,answers,['a','b'])
    assert result['arms']['a']['all']['targets']==3 and result['arms']['a']['all']['returned']==2
    assert result['arms']['a']['S_present/negative']['negative__unavailable']==1
    assert result['arms']['a']['S_present/unavailable']['unavailable__missing']==1
    assert result['arms']['b']['all']['agreement']==2
    assert result['semantic_admission']==0 and not result['bulk_dispatch_allowed']


def test_terminal_file_scoring_refuses_partial_run_and_preserves_format_failures(tmp_path):
    from score_three_state_calibration import score
    def write(p,value):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p
    data=tmp_path/'data';data.mkdir()
    user=dict(instruction='review',ontology={},questions=[dict(id='q')],reference_text='fixture',response_contract={})
    case=dict(request_id='r',semantic_id='q',family='I_present',expected_status='positive')
    for name,value in [('cases.offline.jsonl',case),('requests.jsonl',dict(request_id='r',user=user))]:
        (data/name).write_text(json.dumps(value)+chr(10))
    write(data/'report.json',dict(files={n:sha(data/n) for n in ('cases.offline.jsonl','requests.jsonl')}))
    root=tmp_path/'run';root.mkdir();jobs=[];terminals=[]
    for arm in ('envelope','envelope_three_state'):
        p=root/arm
        messages=[dict(role='system',content='base'+(' addition' if arm=='envelope_three_state' else '')),
                  dict(role='user',content=json.dumps(user))]
        mp=write(p/'messages.private.json',messages)
        cfg=dict(model='alias',max_tokens=4096,stream=False)
        cp=write(p/'config.private.json',cfg)
        write(p/'response/request.json',dict(**cfg,messages=messages))
        decisions=[dict(id='q',status='positive',rationale='Fixture.')]
        # Failed control remains in denominator; no wrapping the array.
        parsed=decisions if arm=='envelope' else dict(decisions=decisions)
        write(p/'response/response.json',dict(choices=[dict(finish_reason='stop',message=dict(content=json.dumps(parsed)))],
                                           usage=dict(prompt_tokens=10,completion_tokens=5,total_tokens=15)))
        result=dict(state='response_received',http_status=200,finish_reasons=['stop'],
                    usage=dict(prompt_tokens=10,completion_tokens=5,total_tokens=15))
        write(p/'response/result.json',result)
        term=dict(arm=arm,request_id='r',exit_code=0,**result)
        write(p/'terminal.json',term);terminals.append(term)
        jobs.append(dict(arm=arm,request_id='r',panel_id='panel',path=str(p),messages_sha256=sha(mp),config_sha256=sha(cp)))
    plan=dict(data_root=str(data),data_report_sha256=sha(data/'report.json'),jobs=jobs,total_requests=2,
              arms=['envelope','envelope_three_state'],directive=' addition',model_identity='synthetic',
              model='alias',max_tokens_per_request=4096)
    write(root/'plan.json',plan);write(root/'launch.json',dict(plan_sha256=sha(root/'plan.json')))
    write(root/'terminal.json',dict(results=terminals[:1]))
    with pytest.raises(ValueError,match='all terminal'):
        score(root,tmp_path/'observed.json')
    assert not (tmp_path/'observed.json').exists()
    write(root/'terminal.json',dict(results=terminals))
    result=score(root,tmp_path/'observed.json')
    assert result['arms']['envelope']['all']['positive__missing']==1
    assert result['arms']['envelope_three_state']['narrow_calibration_gate_passed']
    assert not result['bulk_dispatch_allowed'] and not result['training_dispatch_allowed']
