#!/usr/bin/env python3
"""Separate a semantic proposal, native agreement, and admission without relabelling."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_sha(value):
    return hashlib.sha256((json.dumps(value, sort_keys=True, ensure_ascii=False)+'\n').encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def convert(decision):
    """Preserve a reference review as a proposal, including disagreement with native."""
    if decision.get('optimization_allowed') is not False:
        raise ValueError('only nonadmitted reference reviews can be migrated')
    status = decision['status']
    value, native = decision['proposed_semantic_value'], decision['native_annotated']
    if type(native) is not bool:
        raise ValueError('native annotation membership must be boolean')
    if status == 'unavailable':
        if value is not None:
            raise ValueError('unavailable must not invent a value')
    elif status in ('positive', 'negative', 'conflict'):
        if type(value) is not bool:
            raise ValueError('explicit boolean proposal required')
        if status == 'positive' and value is not True or status == 'negative' and value is not False:
            raise ValueError('status/value mismatch')
        if (status == 'conflict') != (value != native):
            raise ValueError('native disagreement/status mismatch')
    else:
        raise ValueError('unknown legacy review status')
    for field in ('evidence', 'limitation'):
        if not isinstance(decision.get(field), str) or not decision[field].strip():
            raise ValueError('explicit evidence and limitations required')
    return dict(
        schema='jev-semantic-reference-proposal-v1', source_id=decision['source_id'],
        operator='S_present', definition_id=decision['entity_type'],
        metadata_record_sha256=decision['metadata_record_sha256'],
        definition_sha256=decision['definition_sha256'], policy_sha256=decision['policy_sha256'],
        semantic_proposal=dict(value=value, status='unresolved' if value is None else 'proposed',
            evidence=decision['evidence'], limitation=decision['limitation'],
            evidence_modality='reference_text', independent_semantic_review=False),
        native_projection=dict(value=native, agreement='not_assessed' if value is None else
            'agrees' if value == native else 'disagrees', original_annotation_modified=False,
            native_annotation_error_certified=False),
        admission=dict(status='pending', value=None, optimization_allowed=False,
            reason='Reference proposal requires separate task/coverage and adjudication release.'),
        legacy=dict(status=status, decision_sha256=content_sha(decision)),
        actor_input_allowed=False)


def rebind(decisions, bindings):
    lookup = {}
    identities = set()
    converted = []
    for d in decisions:
        identity = (d['source_id'], d['entity_type'])
        if identity in identities:
            raise ValueError('duplicate source/type proposal')
        identities.add(identity)
        proposal = convert(d)
        lookup[content_sha(d)] = proposal
        converted.append(proposal)
    output, seen, covered = [], set(), set()
    for b in bindings:
        if b.get('optimization_allowed') is not False:
            raise ValueError('unexpected binding admission')
        if b['review_sha256'] not in lookup:
            raise ValueError('binding refers to an unknown review')
        p = lookup[b['review_sha256']]
        if (b['source_id'], b['entity_type']) != (p['source_id'], p['definition_id']):
            raise ValueError('binding source/type mismatch')
        identity = (b['source_id'], b['entity_type'], b['recording_id'])
        if identity in seen:
            raise ValueError('duplicate recording binding')
        seen.add(identity)
        covered.add(b['review_sha256'])
        # Original source/audio identities and evidence scope survive unchanged.
        output.append(dict(b, legacy_review_sha256=b['review_sha256'],
            review_sha256=content_sha(p), semantic_proposed_value=p['semantic_proposal']['value'],
            native_agreement=p['native_projection']['agreement'],
            status='proposal_only_not_training_target', admitted_value=None,
            optimization_allowed=False))
    if covered != set(lookup):
        raise ValueError('every proposal needs at least one explicit recording binding')
    return converted, output


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-report', type=Path, required=True)
    p.add_argument('--source-report-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if sha(a.source_report) != a.source_report_sha256:
        raise ValueError('source report changed')
    report = read(a.source_report)
    pins = dict(report['pins'])
    pins[str(a.source_report)] = a.source_report_sha256
    for name, digest in report['files'].items():
        pins[str(a.source_report.parent/name)] = digest
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('dependency changed: '+path)
    proposals, bindings = rebind(rows(a.source_report.parent/'decisions.jsonl'),
                                rows(a.source_report.parent/'recording-bindings.jsonl'))
    a.output.mkdir(parents=True)
    for name, records in [('proposals.jsonl', proposals), ('recording-bindings.jsonl', bindings)]:
        (a.output/name).write_text(''.join(json.dumps(r, sort_keys=True, ensure_ascii=False)+'\n' for r in records))
    summary = dict(schema='jev-semantic-reference-layer-v1', pins=pins,
        implementation_sha256=sha(__file__), proposals=len(proposals), recording_bindings=len(bindings),
        unique_recordings=len({b['recording_id'] for b in bindings}),
        semantic_status_counts=dict(Counter(p['semantic_proposal']['status'] for p in proposals)),
        proposed_value_counts=dict(Counter('null' if p['semantic_proposal']['value'] is None else
            str(p['semantic_proposal']['value']).lower() for p in proposals)),
        native_agreement_counts=dict(Counter(p['native_projection']['agreement'] for p in proposals)),
        admitted_decisions=0, training_dispatch_allowed=False,
        scope='Lossless structural migration of reference proposals; no new semantic judgement, native correction, independent review, or training admission.',
        files={name:sha(a.output/name) for name in ('proposals.jsonl','recording-bindings.jsonl')})
    (a.output/'report.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('pins','files')}, indent=2))


if __name__ == '__main__':
    main()
