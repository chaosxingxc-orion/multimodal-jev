from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/jev'))
from score_semantic_quality_audit import cluster_bound, summarize, load_pass, unique_index
import json
import hashlib


def population():
    coords, p, r, a = [], {}, {}, {}
    for i in range(128):
        for family in ('I_present', 'S_present'):
            c = dict(request_id=family+str(i), semantic_id='predicate', source_group=f'g{i:03}',
                     panel_id=family+'-00', family=family)
            key = c['request_id'], c['semantic_id']
            coords.append(c)
            status = 'positive' if i % 2 else 'negative'
            p[key] = r[key] = status
            a[key] = {'status': status, 'evidence': 'Individually reviewed supplied evidence.'}
    return coords, p, r, a


def test_group_bootstrap_matches_independent_scalar_resampling_and_includes_empty_groups():
    groups = ['a', 'b', 'c', 'd']
    correct = {'a': 7, 'b': 1}
    proposed = {'a': 9, 'b': 2, 'c': 1}
    expected = []
    rng = np.random.default_rng(317)
    zeros = 0
    for _ in range(100):
        selected = [groups[i] for i in rng.integers(0, 4, size=4)]
        den = sum(proposed.get(g, 0) for g in selected)
        if not den:
            zeros += 1
        else:
            expected.append(sum(correct.get(g, 0) for g in selected)/den)
    result = cluster_bound(correct, proposed, groups, replicates=100)
    assert result['zero_denominator_replicates'] == zeros
    assert result['lower95'] == (None if zeros else np.quantile(expected, .05, method='lower'))
    result = cluster_bound({'a': 1}, {'a': 1}, groups, replicates=10000)
    assert not result['estimable'] and result['zero_denominator_replicates'] > 0


def test_complete_reviews_report_numerical_pass_without_admission_and_degenerate_bound():
    coords, p, r, a = population()
    report = summarize(coords, p, r, a)
    assert report['all_required_reviews_present']
    assert all(b['numerical_gate_passed'] and b['zero_observed_errors'] for b in report['family_polarity_bounds'].values())
    assert all(c['both_polarities_observed'] for c in report['definition_coverage'].values())
    assert report['semantic_admission'] == 0 and not report['training_dispatch_allowed']


def test_agreement_without_adjudication_is_not_counted_correct_or_admitted():
    coords, p, r, a = population()
    key = coords[1]['request_id'], coords[1]['semantic_id']
    family = coords[1]['family'];polarity = p[key]
    del a[key]
    report = summarize(coords, p, r, a)
    assert report['pending_adjudication'] == 1
    assert not report['all_required_reviews_present']
    bound = report['family_polarity_bounds'][family+'/'+polarity]
    assert bound['proposed'] == 64 and bound['correct'] == 63


def test_missing_review_retains_proposer_denominator_and_unavailable_is_not_false():
    coords, p, r, a = population()
    c=coords[0];key=c['request_id'],c['semantic_id']
    del r[key]
    other=coords[3];otherkey=other['request_id'],other['semantic_id']
    a[otherkey]={'status':'unavailable','evidence':'Ambiguous role under the supplied rules.'}
    report=summarize(coords,p,r,a)
    assert report['paired_review_missing']==1
    assert report['family_polarity_bounds'][c['family']+'/negative']['proposed']==64
    assert report['family_polarity_bounds'][c['family']+'/negative']['correct']==63
    assert report['by_definition'][other['family']+'/predicate']['adjudicated_unavailable']==1
    assert not report['all_required_reviews_present']


def test_conflicting_review_needs_explicit_adjudication_and_scope_keys_are_checked():
    coords,p,r,a=population()
    c=coords[0];key=c['request_id'],c['semantic_id'];r[key]='positive'
    report=summarize(coords,p,r,a)
    assert report['disagreements']==1
    assert report['by_panel'][c['panel_id']]['paired_negative__positive']==1
    with pytest.raises(ValueError,match='unique'):
        summarize(coords+[c],p,r,a)
    with pytest.raises(ValueError,match='unknown'):
        summarize(coords,{**p,('extra','id'):'negative'},r,a)
    a[key]['evidence']=''
    with pytest.raises(ValueError,match='evidence'):
        summarize(coords,p,r,a)


def test_pass_requires_raw_complete_evidence_and_pinned_inputs(tmp_path):
    def write(name, data):
        path = tmp_path/name
        path.write_text(json.dumps(data))
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    response = dict(decisions=[dict(id='slot', status='negative', rationale='Absent in reference.')])
    templates = {'p': dict(questions=[dict(id='slot')], instruction='Review the reference.',
                          ontology={}, response_contract={})}
    sources = {'s': dict(reference_text='hello')}
    from build_blind_semantic_annotation import render
    prompt = tmp_path/'system.txt';prompt.write_text('Return the specified JSON object.')
    messages = [dict(role='system', content=prompt.read_text()),
                dict(role='user', content=json.dumps(render(templates['p'], sources['s'])))]
    request = write('request.json', messages)
    manifest = write('manifest.json', dict(schema='jev-semantic-quality-input-manifest-v1',
        cohort_report_sha256='cohort', role='proposer', requests=[dict(request_id='r', **request)],
        system_prompt_path=str(prompt), system_prompt_sha256=hashlib.sha256(prompt.read_bytes()).hexdigest()))
    evidence = write('response.json', dict(choices=[dict(finish_reason='stop',
        message=dict(content=json.dumps(response)))]))
    entry = dict(request_id='r', valid=True, response=response, evidence=evidence)
    data = dict(schema='jev-semantic-quality-pass-v1', role='proposer', cohort_report_sha256='cohort',
        reviewer_id='reviewer', complete=True, method='reference_text', model_identity='provider-alias',
        input_manifest_path=manifest['path'], input_manifest_sha256=manifest['sha256'], requests=[entry])
    schedule = {'r': dict(panel_id='p', review_id='s')}
    def run():
        write('pass.json', data)
        return load_pass(tmp_path/'pass.json', 'cohort', schedule, templates, 'proposer', sources)
    _, judgments, transport = run()
    assert judgments == {('r', 'slot'): 'negative'} and transport['p']['fraction'] == 1
    entry['response'] = dict(decisions=[dict(id='slot', status='positive', rationale='Altered.')])
    with pytest.raises(ValueError, match='differs from raw'):
        run()
    entry['response'] = response
    entry['evidence'] = write('response.json', dict(choices=[dict(finish_reason='length',
        message=dict(content=json.dumps(response)))]))
    with pytest.raises(ValueError, match='complete raw'):
        run()
    entry.update(valid=False, response=None, failure='length')
    _, judgments, transport = run()
    assert not judgments and transport['p']['requests'] == 1 and transport['p']['fraction'] == 0
    entry['response'] = response
    with pytest.raises(ValueError, match='partial failed'):
        run()
    entry['response'] = None
    data['complete'] = False
    with pytest.raises(ValueError, match='unfinished'):
        run()
    data['complete'] = True
    write('request.json', {'messages': ['changed']})
    with pytest.raises(ValueError, match='input request changed'):
        run()
    # Even a newly hashed transport file must contain the exact cohort input.
    changed = json.loads(messages[1]['content']);changed['native_gold']='negative'
    messages[1]['content']=json.dumps(changed)
    request = write('request.json', messages)
    envelope=json.loads((tmp_path/'manifest.json').read_text())
    envelope['requests']=[dict(request_id='r', **request)]
    manifest=write('manifest.json', envelope)
    data['input_manifest_sha256']=manifest['sha256']
    with pytest.raises(ValueError, match='differs from frozen cohort'):
        run()


def test_duplicate_cohort_identities_cannot_be_silently_collapsed():
    with pytest.raises(ValueError, match='duplicate'):
        unique_index([{'panel_id': 'p'}, {'panel_id': 'p'}], 'panel_id')
    with pytest.raises(ValueError, match='empty'):
        unique_index([], 'panel_id')


def test_full_cohort_files_flow_into_scoring_with_missing_review_retained(tmp_path):
    from prepare_semantic_quality_audit import prepare_inputs
    from consolidate_intent_presence_references import obj, sha
    from score_semantic_quality_audit import score
    cohort=tmp_path/'cohort';cohort.mkdir()
    sources=[dict(review_id=f's{i:03}',reference_text=f'synthetic fixture {i}') for i in range(128)]
    templates=[]
    for family,count in [('I_present',65),('S_present',55)]:
        for start in range(0,count,8):
            templates.append(dict(panel_id=f'{family}-{start//8:02}',family=family,instruction='Review fixture.',
                ontology={},response_contract={},questions=[dict(id=f'{family}-{i}') for i in range(start,min(start+8,count))]))
    schedule=[dict(request_id=obj([s['review_id'],obj(t)]),review_id=s['review_id'],panel_id=t['panel_id'],
                   template_sha256=obj(t)) for s in sources for t in templates]
    payload={'sources.jsonl':sources,'templates.jsonl':templates,'schedule.jsonl':schedule,
        'bindings.offline.jsonl':[dict(review_id=s['review_id'],bindings=[dict(source_group=s['review_id'])]) for s in sources]}
    for name,values in payload.items():
        (cohort/name).write_text(''.join(json.dumps(v)+chr(10) for v in values))
    report=dict(schema='jev-semantic-quality-audit-cohort-v1',sources=128,groups=128,
        requests_per_pass=2048,decisions_per_pass=15360,files={n:sha(cohort/n) for n in payload})
    def write(path,value):
        path.write_text(json.dumps(value));return path
    write(cohort/'report.json',report)
    prompt=tmp_path/'system.txt';prompt.write_text('Return specified JSON.')
    template_index={t['panel_id']:t for t in templates}
    raws={}
    for t in templates:
        for status in ('positive','negative'):
            response=dict(decisions=[dict(id=q['id'],status=status,rationale='Synthetic reviewed fixture.') for q in t['questions']])
            raw=write(tmp_path/(t['panel_id']+status+'.json'),dict(choices=[dict(finish_reason='stop',
                message=dict(content=json.dumps(response)))]))
            raws[t['panel_id'],status]=(response,dict(path=str(raw),sha256=sha(raw)))
    pass_paths=[]
    for role in ('proposer','blind_reviewer'):
        inputs=tmp_path/role;prepare_inputs(cohort,prompt,role,inputs)
        outcomes=[]
        for i,s in enumerate(schedule):
            status='positive' if int(s['review_id'][1:])%2 else 'negative'
            response,evidence=raws[s['panel_id'],status]
            entry=dict(request_id=s['request_id'],valid=True,response=response,evidence=evidence)
            if role=='blind_reviewer' and i==0:
                entry.update(valid=False,response=None,failure='fixture_transport_failure')
            outcomes.append(entry)
        pass_paths.append(write(tmp_path/(role+'.json'),dict(schema='jev-semantic-quality-pass-v1',role=role,
            cohort_report_sha256=sha(cohort/'report.json'),reviewer_id=role,model_identity='synthetic-fixture',
            complete=True,method='reference_text',input_manifest_path=str(inputs/'manifest.json'),
            input_manifest_sha256=sha(inputs/'manifest.json'),requests=outcomes)))
    adjudications=[]
    for s in schedule:
        status='positive' if int(s['review_id'][1:])%2 else 'negative'
        for q in template_index[s['panel_id']]['questions']:
            adjudications.append(dict(request_id=s['request_id'],semantic_id=q['id'],status=status,evidence='Synthetic fixture.'))
    ad=write(tmp_path/'adjudication.json',dict(schema='jev-semantic-quality-adjudication-v1',
        cohort_report_sha256=sha(cohort/'report.json'),proposer_pass_sha256=sha(pass_paths[0]),
        reviewer_pass_sha256=sha(pass_paths[1]),adjudicator_id='fixture-adjudicator',decisions=adjudications))
    protocol=write(tmp_path/'protocol.json',dict(quality_cohort=dict(report=str(cohort/'report.json')),
        preregistered_quality_gates=dict(transport_complete_fraction_min=.98,minimum_groups_per_family_polarity=30)))
    result=score(cohort,protocol,*pass_paths,ad,tmp_path/'observed.json')
    assert result['coordinates']==15360 and result['groups']==128
    assert result['paired_review_missing']==8 and not result['all_required_reviews_present']
    bound=result['family_polarity_bounds']['I_present/negative']
    assert bound['proposed']==4160 and bound['correct']==4152
    assert result['transport']['blind_reviewer']['I_present-00']['complete']==127
    assert result['semantic_admission']==0 and not result['training_dispatch_allowed']
    assert result['user_inputs_match_frozen_cohort'] and not result['reviewer_independence_certified']
    # The same scorer must accept a real CSV-style human evidence envelope,
    # without manufacturing a Chat Completions response for the reviewer.
    import csv
    from prepare_human_semantic_review import prepare as prepare_human, collect as collect_human
    packet=tmp_path/'human';prepare_human(cohort,packet)
    with (packet/'answers.csv').open(newline='') as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames;answers=list(reader)
    truth={(a['request_id'],a['semantic_id']):a['status'] for a in adjudications}
    for a in answers:
        a.update(status=truth[a['request_id'],a['semantic_id']],rationale='Synthetic human-shaped evidence fixture.')
    with (packet/'answers.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(answers)
    write(packet/'reviewer.json',dict(schema='jev-human-reviewer-identity-v1',reviewer_id='fixture-human',
        method='reference_text',started_at='2026-11-13T00:00:00Z',finished_at='2026-11-13T01:00:00Z',
        prior_exposure='Synthetic fixture only',blind_review_attested=True))
    collect_human(packet,packet/'answers.csv',packet/'reviewer.json',tmp_path/'human-result')
    human_pass=tmp_path/'human-result/pass.json'
    ad_data=json.loads(ad.read_text());ad_data['reviewer_pass_sha256']=sha(human_pass)
    human_ad=write(tmp_path/'human-adjudication.json',ad_data)
    human_result=score(cohort,protocol,pass_paths[0],human_pass,human_ad,tmp_path/'human-observed.json')
    assert human_result['all_required_reviews_present'] and human_result['coordinates']==15360
    assert human_result['identity_claims']['blind_reviewer']['model_identity']=='not_applicable_human_reference_review'
    assert human_result['semantic_admission']==0 and not human_result['reviewer_independence_certified']
