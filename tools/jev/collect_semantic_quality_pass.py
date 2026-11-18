#!/usr/bin/env python3
"""Collect one complete frozen reviewer pass; never dispatch or retry requests."""
import argparse
import json
from pathlib import Path

from build_blind_semantic_annotation import validate_response
from consolidate_intent_presence_references import read, require, sha
from parse_semantic_annotation_response import parse_response_content
from prepare_semantic_quality_audit import load_quality_cohort, unique_records
from score_semantic_quality_audit import load_pass


def pinned_file(evidence):
    require(isinstance(evidence, dict) and set(evidence) == {'path', 'sha256'}, 'pinned evidence required')
    require(sha(evidence['path']) == evidence['sha256'], 'execution evidence changed')
    return read(evidence['path'])


def collect(manifest_path, index_path, output):
    require(not output.exists(), 'output exists')
    manifest, index = read(manifest_path), read(index_path)
    require(manifest.get('schema') == 'jev-semantic-quality-input-manifest-v1', 'invalid input manifest')
    require(index.get('schema') == 'jev-semantic-quality-execution-index-v1', 'invalid execution index')
    require(index.get('input_manifest_sha256') == sha(manifest_path), 'execution/input identity mismatch')
    for key in ('reviewer_id', 'model_identity', 'requested_model'):
        require(isinstance(index.get(key), str) and bool(index[key].strip()), 'explicit reviewer/model identity required')
    cohort = Path(manifest['cohort'])
    _, sources, templates, schedule, _ = load_quality_cohort(cohort)
    cohort_hash = sha(cohort/'report.json')
    require(cohort_hash == manifest['cohort_report_sha256'], 'cohort changed')
    inputs = unique_records(manifest['requests'], 'request_id')
    executions = unique_records(index['requests'], 'request_id')
    require(set(inputs) == set(executions) == set(schedule), 'complete execution index required')
    outcomes = []
    for rid, task in schedule.items():
        job = executions[rid]
        terminal = pinned_file(job['terminal'])
        require(terminal.get('request_id') == rid and type(terminal.get('exit_code')) is int,
                'authoritative terminal required')
        sent, result = pinned_file(job['request']), pinned_file(job['result'])
        require(sha(inputs[rid]['path']) == inputs[rid]['sha256'], 'input request changed')
        require(sent.get('messages') == read(inputs[rid]['path']), 'sent messages differ from frozen inputs')
        require(sent.get('model') == index['requested_model'], 'requested model changed')
        outcome = dict(request_id=rid, valid=False, response=None, evidence=job['result'],
                       failure='transport_or_process_failure')
        if terminal['exit_code'] == 0 and result.get('state') == 'response_received' and result.get('http_status') == 200:
            require('response' in job, 'successful transport requires raw response evidence')
            raw = pinned_file(job['response'])
            require(result.get('usage') == raw.get('usage'), 'response usage differs')
            choices = raw.get('choices')
            require(isinstance(choices, list), 'raw choices missing')
            require(result.get('finish_reasons') == [c.get('finish_reason') for c in choices],
                    'terminal finish reasons differ')
            outcome['evidence'] = job['response']
            outcome['failure'] = 'incomplete_or_multiple_choices'
            if len(choices) == 1 and choices[0].get('finish_reason') == 'stop':
                try:
                    parsed, representation = parse_response_content(choices[0]['message'].get('content', ''))
                    validate_response(parsed, templates[task['panel_id']])
                except (ValueError, TypeError, KeyError) as error:
                    outcome['failure'] = 'invalid_output: ' + str(error)
                else:
                    outcome.update(valid=True, response=parsed, representation=representation)
                    outcome.pop('failure')
        outcomes.append(outcome)
    result = dict(schema='jev-semantic-quality-pass-v1', role=manifest['role'],
        cohort_report_sha256=cohort_hash, reviewer_id=index['reviewer_id'],
        model_identity=index['model_identity'], method='reference_text', complete=True,
        input_manifest_path=str(manifest_path.resolve()), input_manifest_sha256=sha(manifest_path),
        execution_index_path=str(index_path.resolve()), execution_index_sha256=sha(index_path),
        implementation_sha256=sha(__file__), requests=outcomes, semantic_admission=0,
        training_dispatch_allowed=False, independence_certified=False)
    # Validate the exact artifact before publishing it. An unfinished or malformed
    # index cannot leave a pass file that looks complete.
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    require(not temporary.exists(), 'temporary output exists')
    try:
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2)+chr(10))
        load_pass(temporary, cohort_hash, schedule, templates, manifest['role'], sources)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('manifest', 'index', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.manifest, args.index, args.output)
    print(json.dumps(dict(requests=len(result['requests']), valid=sum(r['valid'] for r in result['requests']),
                         semantic_admission=0, training_dispatch_allowed=False)))
