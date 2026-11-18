from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from prepare_semantic_quality_audit import select_groups
from prepare_semantic_quality_audit import load_quality_cohort, prepare_inputs
from consolidate_intent_presence_references import obj, sha
import json

def binding(s,g):
    return dict(source_id=s,review_id='review-'+s,bindings=[dict(source_group=g)])

def test_exclusion_closes_entire_group_and_selection_preserves_group():
    b=[binding('a','g1'),binding('b','g1'),binding('c','g2'),binding('d','g3'),binding('e','g3')]
    result=select_groups(b,{'a'},2,'seed')
    assert {x['source_id'] for x in result}=={'c','d','e'}
    assert result==select_groups(list(reversed(b)),{'a'},2,'seed')

def test_insufficient_or_unbound_exclusion_cannot_silently_shrink():
    b=[binding('a','g1'),binding('b','g1')]
    with pytest.raises(ValueError,match='insufficient'): select_groups(b,{'a'},1,'seed')
    with pytest.raises(ValueError,match='outside'): select_groups(b,{'z'},1,'seed')

def test_source_ambiguity_and_duplicate_rejected():
    b=binding('a','g1')
    with pytest.raises(ValueError,match='duplicate'): select_groups([b,b],set(),1,'seed')
    b['bindings'].append(dict(source_group='g2'))
    with pytest.raises(ValueError,match='one explicit'): select_groups([b],set(),1,'seed')


def fixture_cohort(path):
    path.mkdir()
    sources=[dict(review_id='s'+str(i),reference_text='reference '+str(i),native_gold='must not enter wire') for i in range(2)]
    templates=[dict(panel_id='p'+str(i),family='I_present',questions=[dict(id='q'+str(i))],
                    instruction='Review',ontology={},response_contract={}) for i in range(2)]
    schedule=[dict(request_id=obj([s['review_id'],obj(t)]),review_id=s['review_id'],
                   panel_id=t['panel_id'],template_sha256=obj(t)) for s in sources for t in templates]
    data={'sources.jsonl':sources,'templates.jsonl':templates,'schedule.jsonl':schedule,
          'bindings.offline.jsonl':[dict(review_id=s['review_id'],bindings=[dict(source_group=s['review_id'])]) for s in sources]}
    for name, values in data.items():
        (path/name).write_text(''.join(json.dumps(v)+chr(10) for v in values))
    report=dict(schema='jev-semantic-quality-audit-cohort-v1',sources=2,groups=2,
                requests_per_pass=4,decisions_per_pass=4,files={n:sha(path/n) for n in data})
    (path/'report.json').write_text(json.dumps(report))
    return report, data


def test_input_freeze_projects_exact_full_schedule_and_excludes_offline_labels(tmp_path):
    cohort=tmp_path/'cohort';fixture_cohort(cohort)
    prompt=tmp_path/'prompt.txt';prompt.write_text('Return JSON.')
    manifest=prepare_inputs(cohort,prompt,'blind_reviewer',tmp_path/'inputs')
    assert len(manifest['requests'])==4 and not manifest['API_sent']
    for request in manifest['requests']:
        assert sha(request['path'])==request['sha256']
        messages=json.loads(Path(request['path']).read_text())
        assert messages[0]=={'role':'system','content':'Return JSON.'}
        user=json.loads(messages[1]['content'])
        assert set(user)=={'instruction','ontology','questions','reference_text','response_contract'}
        assert 'native_gold' not in user
    with pytest.raises(ValueError,match='output exists'):
        prepare_inputs(cohort,prompt,'blind_reviewer',tmp_path/'inputs')


def test_cohort_coverage_cannot_be_replaced_by_aggregate_counts(tmp_path):
    cohort=tmp_path/'cohort';report,data=fixture_cohort(cohort)
    load_quality_cohort(cohort)
    # Remove one source/panel pair and re-pin counts: coverage still must fail.
    data['schedule.jsonl'].pop()
    path=cohort/'schedule.jsonl';path.write_text(''.join(json.dumps(v)+chr(10) for v in data['schedule.jsonl']))
    report['files']['schedule.jsonl']=sha(path);report['requests_per_pass']=3;report['decisions_per_pass']=3
    (cohort/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='incomplete source/panel'):
        load_quality_cohort(cohort)


def fixture_execution(tmp_path):
    from collect_semantic_quality_pass import collect
    cohort=tmp_path/'cohort';fixture_cohort(cohort)
    prompt=tmp_path/'prompt.txt';prompt.write_text('Return JSON.')
    inputs=tmp_path/'inputs';manifest=prepare_inputs(cohort,prompt,'proposer',inputs)
    jobs=[]
    def pin(name, value):
        path=tmp_path/name;path.write_text(json.dumps(value))
        return dict(path=str(path),sha256=sha(path))
    for i,r in enumerate(manifest['requests']):
        messages=json.loads(Path(r['path']).read_text())
        q=json.loads(messages[1]['content'])['questions'][0]['id']
        decision=dict(id=q,status='negative',rationale='Absent from supplied text.')
        parsed=[decision] if i==1 else dict(decisions=[decision])
        finish='length' if i==2 else 'stop'
        raw=dict(choices=[dict(finish_reason=finish,message=dict(content=json.dumps(parsed)))],usage={'total_tokens':10})
        result=dict(state='response_received',http_status=200,finish_reasons=[finish],usage=raw['usage'])
        if i==3:result=dict(state='http_error',http_status=503,usage=None)
        jobs.append(dict(request_id=r['request_id'],
            terminal=pin(f'terminal-{i}.json',dict(request_id=r['request_id'],exit_code=1 if i==3 else 0)),
            request=pin(f'request-{i}.json',dict(model='fixed-alias',messages=messages)),
            result=pin(f'result-{i}.json',result),response=pin(f'raw-{i}.json',raw)))
    index=dict(schema='jev-semantic-quality-execution-index-v1',input_manifest_sha256=sha(inputs/'manifest.json'),
               reviewer_id='reviewer-a',model_identity='provider-alias-only',requested_model='fixed-alias',requests=jobs)
    index_path=tmp_path/'index.json';index_path.write_text(json.dumps(index))
    return inputs/'manifest.json',index_path,index


def test_collector_retains_all_transport_format_and_truncation_failures(tmp_path):
    from collect_semantic_quality_pass import collect
    manifest,index_path,index=fixture_execution(tmp_path)
    result=collect(manifest,index_path,tmp_path/'pass.json')
    assert len(result['requests'])==4 and sum(r['valid'] for r in result['requests'])==1
    assert all(r['response'] is None for r in result['requests'] if not r['valid'])
    assert result['semantic_admission']==0 and not result['training_dispatch_allowed']
    assert 'invalid_output' in result['requests'][1]['failure']
    assert result['requests'][2]['failure']=='incomplete_or_multiple_choices'
    assert result['requests'][3]['failure']=='transport_or_process_failure'
    with pytest.raises(ValueError,match='output exists'):
        collect(manifest,index_path,tmp_path/'pass.json')


def test_collector_rejects_incomplete_terminal_index_without_publishing(tmp_path):
    from collect_semantic_quality_pass import collect
    manifest,index_path,index=fixture_execution(tmp_path)
    index['requests'].pop();index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError,match='complete execution index'):
        collect(manifest,index_path,tmp_path/'pass.json')
    assert not (tmp_path/'pass.json').exists()


def test_collector_rejects_changed_sent_input_even_with_updated_hash(tmp_path):
    from collect_semantic_quality_pass import collect
    manifest,index_path,index=fixture_execution(tmp_path)
    evidence=index['requests'][0]['request'];path=Path(evidence['path'])
    request=json.loads(path.read_text());request['messages'].append(dict(role='user',content='Use a leaked answer.'))
    path.write_text(json.dumps(request));evidence['sha256']=sha(path);index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError,match='sent messages differ'):
        collect(manifest,index_path,tmp_path/'pass.json')
    assert not (tmp_path/'pass.json').exists()
