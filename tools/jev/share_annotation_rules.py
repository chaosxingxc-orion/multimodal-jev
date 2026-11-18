#!/usr/bin/env python3
"""Losslessly share task rules in an existing full annotation workpack."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

from build_blind_semantic_annotation import render
from consolidate_intent_presence_references import obj, read, require, rows, sha
from probe_slurp_panel_context import share_slot_rules, expand_slot_rules


def compact_template(template):
    compact = deepcopy(template)
    compact['ontology'] = share_slot_rules(template['ontology'])
    require(expand_slot_rules(compact['ontology']) == template['ontology'], 'rule sharing changed ontology')
    # Question IDs, instructions and response schema stay byte-for-byte equivalent.
    require({k: v for k, v in compact.items() if k != 'ontology'} ==
            {k: v for k, v in template.items() if k != 'ontology'}, 'non-ontology fields changed')
    return compact


def json_bytes(value):
    return len(json.dumps(value, ensure_ascii=False).encode())


def measure(templates, sources):
    """Exact sum for render's single variable text string; no token approximation."""
    require(bool(sources), 'sources required')
    text_sizes = [json_bytes(s['reference_text']) for s in sources]
    result = {}
    for t in templates:
        overhead = json_bytes(render(t, {'reference_text': ''})) - json_bytes('')
        result[t['panel_id']] = dict(min_json_bytes=overhead + min(text_sizes),
            max_json_bytes=overhead + max(text_sizes),
            total_json_bytes=len(sources)*overhead + sum(text_sizes))
    return dict(panels=result, total_json_bytes=sum(v['total_json_bytes'] for v in result.values()))


def build(source, output):
    require(not output.exists(), 'output exists')
    report_path = source / 'report.json'
    prior = read(report_path)
    require(prior['execution']['API_sent'] is False and prior['execution']['semantic_admission'] == 0,
            'unexecuted reference-only workpack required')
    for p, h in prior['pins'].items():
        require(sha(p) == h, 'source dependency changed: ' + p)
    for p, h in prior['files'].items():
        require(sha(source / p) == h, 'source artifact changed: ' + p)
    templates = rows(source / 'templates.jsonl')
    sources = rows(source / 'sources.jsonl')
    compact = [compact_template(t) for t in templates]
    sizes = dict(literal=measure(templates, sources), shared=measure(compact, sources))
    output.mkdir(parents=True)
    for name in ('sources.jsonl', 'bindings.offline.jsonl'):
        shutil.copyfile(source / name, output / name)
    with (output / 'templates.jsonl').open('x') as stream:
        for t in compact:
            stream.write(json.dumps(t, ensure_ascii=False, sort_keys=True)+'\n')
    hashes = {t['panel_id']: obj(t) for t in compact}
    with (output / 'schedule.jsonl').open('x') as stream:
        for s in sources:
            for t in compact:
                h = hashes[t['panel_id']]
                row = dict(request_id=obj([s['review_id'], h]), review_id=s['review_id'],
                           panel_id=t['panel_id'], template_sha256=h)
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n')
    summary = {k: v for k, v in prior['summary'].items() if k != 'max_request_json_bytes_by_panel'}
    summary['max_request_json_bytes_by_panel'] = {k: v['max_json_bytes'] for k, v in sizes['shared']['panels'].items()}
    result = dict(schema='slurp-blind-semantic-annotation-shared-workpack-v1',
        source_report=str(report_path), source_report_sha256=sha(report_path),
        pins={**prior['pins'], str(report_path): sha(report_path),
              'tools/jev/probe_slurp_panel_context.py': sha('tools/jev/probe_slurp_panel_context.py'),
              'tools/jev/build_blind_semantic_annotation.py': sha('tools/jev/build_blind_semantic_annotation.py')},
        files={n: sha(output / n) for n in prior['files']}, summary=summary,
        input_size_measurement=sizes, implementation_sha256=sha(__file__), execution=prior['execution'],
        independence=prior['independence'],
        scope='Exact rule-string sharing and full-population schedule only. Every field/list position can be restored. Does not establish model comprehension equivalence, provider context limits, token count, cost, labels or training admission.')
    (output / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return dict(summary=summary, literal_bytes=sizes['literal']['total_json_bytes'],
                shared_bytes=sizes['shared']['total_json_bytes'])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.source, a.output), indent=2))
