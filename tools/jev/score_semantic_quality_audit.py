#!/usr/bin/env python3
"""Score paired semantic reviews under the frozen source-group quality policy.

This computes evidence only. It never certifies reviewer independence, admits
labels, dispatches API calls, or changes the training population.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from build_blind_semantic_annotation import render, validate_response
from consolidate_intent_presence_references import obj, read, require, rows, sha
from parse_semantic_annotation_response import parse_response_content
from prepare_semantic_quality_audit import load_quality_cohort

STATUSES = ('positive', 'negative', 'unavailable')


def unique_index(records, key):
    result = {}
    for record in records:
        require(isinstance(record, dict) and key in record, 'missing identity: ' + key)
        identity = record[key]
        require(isinstance(identity, str) and bool(identity) and identity not in result,
                'duplicate or invalid identity: ' + key)
        result[identity] = record
    require(bool(result), 'empty identity collection: ' + key)
    return result


def cluster_bound(correct, proposed, group_order, replicates=10000, seed=317):
    """Resample all source groups, including groups with no proposed polarity."""
    import numpy as np
    require(len(group_order) == len(set(group_order)) and bool(group_order), 'unique groups required')
    require(set(correct) <= set(group_order) and set(proposed) <= set(group_order), 'unknown group')
    require(type(replicates) is int and replicates > 0, 'positive replicate count required')
    numerator = np.array([correct.get(g, 0) for g in group_order], dtype=np.int64)
    denominator = np.array([proposed.get(g, 0) for g in group_order], dtype=np.int64)
    require(bool(((numerator >= 0) & (denominator >= numerator)).all()), 'invalid correctness counts')
    draws = np.random.default_rng(seed).integers(0, len(group_order), size=(replicates, len(group_order)))
    den = denominator[draws].sum(axis=1)
    num = numerator[draws].sum(axis=1)
    zero = int((den == 0).sum())
    bound = None if zero else float(np.quantile(num / den, .05, method='lower'))
    return dict(correct=int(numerator.sum()), proposed=int(denominator.sum()),
                groups_with_proposals=int((denominator > 0).sum()), total_groups=len(group_order),
                bootstrap_replicates=replicates, seed=seed, zero_denominator_replicates=zero,
                lower95=bound, estimable=bound is not None,
                zero_observed_errors=bool(denominator.sum() > 0 and numerator.sum() == denominator.sum()))


def summarize(coordinates, proposer, reviewer, adjudications):
    """Missing pass or adjudication is explicit; agreement alone is not truth."""
    keys = [(c['request_id'], c['semantic_id']) for c in coordinates]
    require(len(keys) == len(set(keys)) and bool(keys), 'unique nonempty coordinates required')
    allowed = set(keys)
    require(set(proposer) <= allowed and set(reviewer) <= allowed and set(adjudications) <= allowed, 'unknown coordinate')
    for judgments in (proposer, reviewer):
        require(all(v in STATUSES for v in judgments.values()), 'invalid judgment status')
    for a in adjudications.values():
        require(isinstance(a, dict) and a.get('status') in STATUSES
                and isinstance(a.get('evidence'), str) and bool(a['evidence'].strip()), 'adjudication evidence required')
    group_order = sorted({c['source_group'] for c in coordinates})
    correct, proposed = defaultdict(Counter), defaultdict(Counter)
    by_definition, by_family, by_panel = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    review_missing = adjudication_missing = disagreements = 0
    for c in coordinates:
        key = c['request_id'], c['semantic_id']
        p, r, a = proposer.get(key), reviewer.get(key), adjudications.get(key)
        complete = p is not None and r is not None
        needs_adjudication = complete  # Includes agreements used to assess correctness.
        pending = needs_adjudication and a is None
        disagreement = complete and p != r
        review_missing += not complete
        adjudication_missing += pending
        disagreements += disagreement
        for counter in (by_definition[c['family'] + '/' + c['semantic_id']], by_family[c['family']], by_panel[c['panel_id']]):
            counter['coordinates'] += 1
            counter['proposed_' + (p or 'missing')] += 1
            counter['reviewed_' + (r or 'missing')] += 1
            counter['adjudicated_' + (a['status'] if a else 'missing')] += 1
            counter['disagreements'] += disagreement
            counter['pending_adjudication'] += pending
            counter['paired_' + (p or 'missing') + '__' + (r or 'missing')] += 1
            if complete and a:
                counter['reviewed_adjudicated_' + a['status']] += 1
        if p in ('positive', 'negative'):
            bucket = c['family'], p
            proposed[bucket][c['source_group']] += 1
            if complete and a and a['status'] == p:
                correct[bucket][c['source_group']] += 1
    bounds = {}
    for family in sorted({c['family'] for c in coordinates}):
        for polarity in ('positive', 'negative'):
            b = cluster_bound(correct[family, polarity], proposed[family, polarity], group_order)
            b['minimum_groups'] = 30
            b['threshold'] = .95
            b['numerical_gate_passed'] = bool(b['estimable'] and b['lower95'] >= .95 and b['groups_with_proposals'] >= 30)
            bounds[family + '/' + polarity] = b
    coverage = {key: dict(positive=v['reviewed_adjudicated_positive'], negative=v['reviewed_adjudicated_negative'],
                         both_polarities_observed=v['reviewed_adjudicated_positive'] > 0 and v['reviewed_adjudicated_negative'] > 0)
                for key, v in sorted(by_definition.items())}
    return dict(coordinates=len(coordinates), groups=len(group_order), paired_review_missing=review_missing,
                pending_adjudication=adjudication_missing, disagreements=disagreements,
                all_required_reviews_present=not review_missing and not adjudication_missing,
                family_polarity_bounds=bounds, definition_coverage=coverage,
                by_family={k: dict(v) for k, v in sorted(by_family.items())},
                by_panel={k: dict(v) for k, v in sorted(by_panel.items())},
                by_definition={k: dict(v) for k, v in sorted(by_definition.items())},
                semantic_admission=0, training_dispatch_allowed=False,
                scope='Numerical evidence only. Independent identity, adjudication quality, type coverage adequacy and full-population release remain separate; no automatic admission.')


def load_pass(path, cohort_hash, schedule, templates, role, sources):
    data = read(path)
    if data.get('schema') == 'jev-human-semantic-review-pass-v1':
        from prepare_human_semantic_review import load_human_pass
        return load_human_pass(path, cohort_hash, schedule, role)
    require(data.get('schema') == 'jev-semantic-quality-pass-v1' and data.get('role') == role, 'invalid review pass')
    require(data.get('cohort_report_sha256') == cohort_hash, 'changed review cohort')
    require(isinstance(data.get('reviewer_id'), str) and bool(data['reviewer_id'].strip()), 'reviewer identity required')
    require(data.get('complete') is True, 'unfinished pass cannot be scored')
    require(data.get('method') == 'reference_text', 'reference review method required')
    for field in ('model_identity', 'input_manifest_sha256'):
        require(isinstance(data.get(field), str) and bool(data[field].strip()), 'pinned pass identity required')
    manifest_path = data.get('input_manifest_path')
    require(isinstance(manifest_path, str) and bool(manifest_path), 'input manifest path required')
    require(sha(manifest_path) == data['input_manifest_sha256'], 'input manifest changed')
    manifest = read(manifest_path)
    require(manifest.get('schema') == 'jev-semantic-quality-input-manifest-v1'
            and manifest.get('cohort_report_sha256') == cohort_hash
            and manifest.get('role') == role, 'input manifest identity mismatch')
    inputs = unique_index(manifest['requests'], 'request_id')
    require(set(inputs) == set(schedule), 'input manifest schedule mismatch')
    require(sha(manifest['system_prompt_path']) == manifest['system_prompt_sha256'], 'system prompt changed')
    prompt = Path(manifest['system_prompt_path']).read_text()
    require(bool(prompt.strip()), 'empty system prompt')
    for rid, request in inputs.items():
        require(sha(request['path']) == request['sha256'], 'input request changed')
        messages = read(request['path'])
        require(isinstance(messages, list) and len(messages) == 2
                and all(isinstance(m, dict) and set(m) == {'role', 'content'} for m in messages),
                'exact two-message review input required')
        require(messages[0] == dict(role='system', content=prompt)
                and messages[1]['role'] == 'user', 'review input roles changed')
        task = schedule[rid]
        expected = render(templates[task['panel_id']], sources[task['review_id']])
        actual, _ = parse_response_content(messages[1]['content'])
        require(actual == expected, 'review input differs from frozen cohort')
    outcomes = data.get('requests')
    require(isinstance(outcomes, list) and len(outcomes) == len(schedule), 'all scheduled terminal requests required')
    indexed = {}
    for entry in outcomes:
        rid = entry.get('request_id')
        require(rid in schedule and rid not in indexed, 'duplicate or unknown terminal request')
        indexed[rid] = entry
    judgments, panels = {}, defaultdict(Counter)
    for rid, task in schedule.items():
        entry = indexed[rid]
        require(type(entry.get('valid')) is bool, 'explicit request validity required')
        panels[task['panel_id']]['requests'] += 1
        evidence = entry.get('evidence')
        require(isinstance(evidence, dict) and set(evidence) == {'path', 'sha256'}, 'raw request evidence required')
        require(sha(evidence['path']) == evidence['sha256'], 'request evidence changed')
        if entry['valid']:
            raw = read(evidence['path'])
            choices = raw.get('choices')
            require(isinstance(choices, list) and len(choices) == 1
                    and choices[0].get('finish_reason') == 'stop', 'complete raw response required')
            parsed, _ = parse_response_content(choices[0]['message'].get('content', ''))
            validate_response(parsed, templates[task['panel_id']])
            require(parsed == entry.get('response'), 'normalized response differs from raw evidence')
            panels[task['panel_id']]['complete'] += 1
            for decision in entry['response']['decisions']:
                judgments[rid, decision['id']] = decision['status']
        else:
            require(isinstance(entry.get('failure'), str) and bool(entry['failure'].strip()), 'explicit failure required')
            require(entry.get('response') is None, 'partial failed response cannot supply labels')
    transport = {}
    for panel, counts in sorted(panels.items()):
        complete, total = counts['complete'], counts['requests']
        transport[panel] = dict(complete=complete, requests=total, fraction=complete/total,
                                threshold=.98, numerical_gate_passed=complete/total >= .98)
    return data, judgments, transport


def score(cohort, protocol, proposer_path, reviewer_path, adjudication_path, output):
    require(not output.exists(), 'output exists')
    p = read(protocol)
    require(p['quality_cohort']['report'] == str(cohort/'report.json'), 'protocol cohort mismatch')
    gates = p['preregistered_quality_gates']
    require(gates['transport_complete_fraction_min'] == .98 and gates['minimum_groups_per_family_polarity'] == 30,
            'unsupported protocol revision')
    report, sources, templates, schedule, bindings = load_quality_cohort(cohort)
    coordinates = []
    for rid, s in schedule.items():
        t = templates[s['panel_id']]
        require(obj(t) == s['template_sha256'], 'template binding changed')
        groups = {r['source_group'] for r in bindings[s['review_id']]['bindings']}
        require(len(groups) == 1, 'source group ambiguous')
        for q in t['questions']:
            coordinates.append(dict(request_id=rid, semantic_id=q['id'], source_group=next(iter(groups)),
                                    family=t['family'], panel_id=t['panel_id']))
    require(len(coordinates) == report['decisions_per_pass'] == 15360, 'full fixed cohort required')
    require(len({c['source_group'] for c in coordinates}) == 128, 'all128 source groups required')
    cohort_hash = sha(cohort/'report.json')
    pd, prop, pt = load_pass(proposer_path, cohort_hash, schedule, templates, 'proposer', sources)
    rd, rev, rt = load_pass(reviewer_path, cohort_hash, schedule, templates, 'blind_reviewer', sources)
    require(pd['reviewer_id'] != rd['reviewer_id'], 'separate reviewer identities required')
    ad = read(adjudication_path)
    require(ad.get('schema') == 'jev-semantic-quality-adjudication-v1' and ad.get('cohort_report_sha256') == cohort_hash,
            'invalid adjudication envelope')
    require(ad.get('proposer_pass_sha256') == sha(proposer_path) and ad.get('reviewer_pass_sha256') == sha(reviewer_path),
            'adjudication bound to different passes')
    require(isinstance(ad.get('adjudicator_id'), str) and bool(ad['adjudicator_id'].strip()), 'adjudicator identity required')
    adjudications = {}
    for entry in ad['decisions']:
        key = entry['request_id'], entry['semantic_id']
        require(key not in adjudications, 'duplicate adjudication')
        adjudications[key] = entry
    observed = summarize(coordinates, prop, rev, adjudications)
    observed.update(schema='jev-semantic-quality-observed-v1', protocol_sha256=sha(protocol), cohort_report_sha256=cohort_hash,
        proposer_pass_sha256=sha(proposer_path), reviewer_pass_sha256=sha(reviewer_path),
        adjudication_sha256=sha(adjudication_path), implementation_sha256=sha(__file__),
        transport=dict(proposer=pt, blind_reviewer=rt),
        identity_claims={role: {k:d[k] for k in ('reviewer_id','model_identity','input_manifest_sha256')} for role,d in (('proposer',pd),('blind_reviewer',rd))},
        reviewer_independence_certified=False,
        input_semantic_blinding_certified=False,
        user_inputs_match_frozen_cohort=True,
        transport_authenticity_certified=False,
        unresolved_definition_conflicts=ad.get('unresolved_definition_conflicts', 'not_reported'))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(observed,ensure_ascii=False,indent=2)+chr(10))
    return observed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('cohort', 'protocol', 'proposer', 'reviewer', 'adjudication', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = score(args.cohort,args.protocol,args.proposer,args.reviewer,args.adjudication,args.output)
    print(json.dumps({k:result[k] for k in ('coordinates','paired_review_missing','pending_adjudication','semantic_admission')},indent=2))
