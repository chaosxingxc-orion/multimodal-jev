import csv
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from prepare_human_semantic_review import prepare,collect
from score_semantic_quality_audit import load_pass
from prepare_semantic_quality_audit import load_quality_cohort
from consolidate_intent_presence_references import sha
from test_semantic_quality_audit import fixture_cohort


def submission(tmp_path):
    cohort=tmp_path/'cohort';fixture_cohort(cohort)
    packet=tmp_path/'packet';prepare(cohort,packet)
    identity=packet/'reviewer.json';who=json.loads(identity.read_text())
    who.update(reviewer_id='synthetic-reviewer',started_at='2026-11-13T00:00:00Z',
        finished_at='2026-11-13T01:00:00Z',prior_exposure='synthetic test only',blind_review_attested=True)
    identity.write_text(json.dumps(who))
    return cohort,packet,identity


def fill(packet):
    path=packet/'answers.csv'
    with path.open(newline='') as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames;records=list(reader)
    for i,row in enumerate(records):
        row.update(status=['positive','negative','unavailable'][i%3],rationale='Synthetic evidence, not a real review.')
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(records)


def test_blank_review_cannot_be_promoted_and_packet_excludes_offline_labels(tmp_path):
    cohort,packet,identity=submission(tmp_path)
    assert 'native_gold' not in (packet/'sources.json').read_text()
    assert not (packet/'bindings.offline.jsonl').exists()
    with pytest.raises(ValueError,match='unfinished'):
        collect(packet,packet/'answers.csv',identity,tmp_path/'imported')
    assert not (tmp_path/'imported').exists()


def test_complete_csv_roundtrip_and_tampered_normalized_answer_rejected(tmp_path):
    cohort,packet,identity=submission(tmp_path);fill(packet)
    output=tmp_path/'imported';result=collect(packet,packet/'answers.csv',identity,output)
    assert result['semantic_admission']==0 and not result['independence_certified']
    _,sources,templates,schedule,_=load_quality_cohort(cohort)
    args=(sha(cohort/'report.json'),schedule,templates,'blind_reviewer',sources)
    data,judgments,transport=load_pass(output/'pass.json',*args)
    assert len(judgments)==4 and set(judgments.values())=={'positive','negative','unavailable'}
    assert data['model_identity']=='not_applicable_human_reference_review'
    assert all(p['complete']==2 for p in transport.values())
    result['requests'][0]['response']['decisions'][0]['status']='negative'
    (output/'pass.json').write_text(json.dumps(result))
    with pytest.raises(ValueError,match='original CSV'):
        load_pass(output/'pass.json',*args)


def test_changed_rules_and_duplicate_rows_rejected(tmp_path):
    cohort,packet,identity=submission(tmp_path);fill(packet)
    with (packet/'answers.csv').open() as stream:lines=stream.readlines()
    (packet/'answers.csv').write_text(''.join(lines+[lines[1]]))
    with pytest.raises(ValueError,match='duplicate'):
        collect(packet,packet/'answers.csv',identity,tmp_path/'imported')
    (packet/'answers.csv').write_text(''.join(lines))
    (packet/'p0.rules.json').write_text('{}')
    with pytest.raises(ValueError,match='material changed'):
        collect(packet,packet/'answers.csv',identity,tmp_path/'imported')
