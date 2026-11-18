from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/jev'))
from build_intent_teacher_calibration import select
from score_intent_teacher_calibration import score, summarize
from consolidate_intent_presence_references import sha


def assessment(sid, status):
    return dict(incompatible_fields=[], optimization_allowed=False,
                decision=dict(source_id=sid, semantic_id='a', reference=sid, status=status))


def test_order_invariance_and_duplicates_keep_evidence_not_extra_weight():
    a = [assessment('p', 'positive'), assessment('n', 'negative'), assessment('u', 'unavailable')]
    a.append(deepcopy(a[0]))
    selected = select(a, {'a'})
    assert selected == select(list(reversed(a)), {'a'})
    assert len(selected) == 3 and len(selected[0]['assessment_sha256']) == 2


def test_conflict_cannot_be_voted_or_replaced_by_latest():
    a = [assessment('p', 'positive'), assessment('p', 'positive'), assessment('p', 'negative'), assessment('n', 'negative')]
    with pytest.raises(ValueError, match='missing polarity'):
        select(a, {'a'})


def test_unavailable_cannot_supply_negative_and_stale_rejected():
    a = [assessment('p', 'positive'), assessment('u', 'unavailable')]
    with pytest.raises(ValueError, match='missing polarity'):
        select(a, {'a'})
    a.append(assessment('n', 'negative'))
    a[0]['incompatible_fields'] = ['card']
    with pytest.raises(ValueError, match='stale'):
        select(a, {'a'})


def test_ambiguity_limit_does_not_change_polarity_targets():
    a = [assessment('p', 'positive'), assessment('n', 'negative')]
    a.extend(assessment(str(i), 'unavailable') for i in range(30))
    assert len(select(a, {'a'})) == 18
    assert select(a, {'a'}, 0) == select(a, {'a'})[:2]


def test_failed_requests_stay_in_denominator_unavailable_is_not_negative():
    cases = [dict(request_id=str(i), semantic_id='a', expected_status=s)
             for i, s in enumerate(['positive','negative','negative','unavailable'])]
    metrics, details = summarize(cases, {'0':{'a':'positive'},'1':{'a':'unavailable'},'3':{'a':'unavailable'}})
    assert metrics['known']['targets']==3
    assert metrics['known']['agreement_all_targets']==1/3
    assert metrics['known']['agreement_returned_only']==1/2
    assert metrics['negative']['agreements']==0
    assert metrics['known']['failures']==1
    assert metrics['ambiguous']['agreements']==1
    assert details[2]['predicted_status'] is None


def test_other_panel_proposals_are_not_gold_scored():
    cases=[dict(request_id='r',semantic_id='a',expected_status='positive')]
    metrics,_=summarize(cases,{'r':{'a':'negative','b':'positive'}})
    assert metrics['known']['targets']==1 and metrics['known']['agreements']==0


def scoring_fixture(tmp_path, finish='stop', content=None):
    packet=tmp_path/'packet';run=tmp_path/'run';request=run/'r'
    packet.mkdir();(request/'response').mkdir(parents=True)
    def write(path, value):
        path.write_text(json.dumps(value)+'\n')
    wire=dict(instruction='Judge supplied data',ontology={},reference_text='hello',
              questions=[{'id':'a'},{'id':'b'}],response_contract={})
    write(packet/'requests.jsonl',dict(request_id='r',panel_id='p',template_sha256='template',wire=wire))
    write(packet/'cases.offline.jsonl',dict(request_id='r',semantic_id='a',expected_status='positive'))
    write(packet/'report.json',dict(files={name:sha(packet/name) for name in ('requests.jsonl','cases.offline.jsonl')}))
    write(run/'plan.json',dict(packet_report_sha256=sha(packet/'report.json'),executor_sha256='executor',max_tokens_per_request=4096))
    messages=[dict(role='system',content='annotator'),dict(role='user',content=json.dumps(wire))]
    write(request/'messages.private.json',messages)
    write(request/'config.private.json',dict(max_tokens=4096,thinking={'type':'enabled'}))
    write(request/'intent.json',dict(request_id='r',template_sha256='template',executor_sha256='executor',
          messages_sha256=sha(request/'messages.private.json'),config_sha256=sha(request/'config.private.json')))
    terminal=dict(request_id='r',state='response_received',exit_code=0)
    write(request/'terminal.json',terminal);write(run/'terminal.json',dict(results=[terminal]))
    write(request/'response/result.json',dict(state='response_received',http_status=200,finish_reasons=[finish],
          usage={'prompt_tokens':30,'completion_tokens':10,'total_tokens':40}))
    write(request/'response/request.json',dict(messages=messages,max_tokens=4096))
    if content is None:
        content=json.dumps(dict(decisions=[dict(id=k,status='positive',rationale='reference evidence') for k in ('a','b')]))
    write(request/'response/response.json',dict(choices=[dict(finish_reason=finish,message={'content':content})]))
    return packet,run,tmp_path/'observed'


def test_complete_run_preserves_extra_proposals_without_scoring_them(tmp_path):
    packet,run,output=scoring_fixture(tmp_path)
    result=score(packet,run,output)
    assert result['valid_requests']==1 and result['metrics']['known']['targets']==1
    assert result['metrics']['known']['agreements']==1
    proposals=json.loads((output/'proposals.offline.jsonl').read_text())
    assert len(proposals['decisions'])==2 and not proposals['optimization_allowed']
    assert result['semantic_admission']==0 and not result['training_dispatch_allowed']


@pytest.mark.parametrize('finish,content',[
    ('length', None), ('stop',''), ('stop','{"decisions":[{"id":"a","status":"positive","rationale":"e"}]}')])
def test_complete_transport_does_not_admit_truncated_or_partial_panel(tmp_path,finish,content):
    packet,run,output=scoring_fixture(tmp_path,finish,content)
    result=score(packet,run,output)
    assert result['valid_requests']==0
    assert result['metrics']['known']['targets']==result['metrics']['known']['failures']==1
    assert result['metrics']['known']['agreement_all_targets']==0
    assert (output/'proposals.offline.jsonl').read_text()==''


def test_running_request_is_not_counted_as_failed(tmp_path):
    packet,run,output=scoring_fixture(tmp_path)
    (run/'terminal.json').unlink()
    with pytest.raises(FileNotFoundError):
        score(packet,run,output)
    assert not output.exists()


def test_sent_wire_mismatch_aborts_scoring(tmp_path):
    packet,run,output=scoring_fixture(tmp_path)
    path=run/'r/response/request.json'
    wire=json.loads(path.read_text());wire['messages'][1]['content']='changed actual request'
    path.write_text(json.dumps(wire))
    with pytest.raises(ValueError,match='sent request changed'):
        score(packet,run,output)
    assert not output.exists()
