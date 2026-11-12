#!/usr/bin/env python3
"""Diagnose authenticated fixed initial/final predictions without rerunning models."""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.span_failure_analysis import diagnose
from multimodal_jev.span_rule_population import configured_rule_rows


def analyze(root, verified_comparison):
    report = json.loads(verified_comparison.read_text())
    if report.get('passed') is not True: raise ValueError('verified full comparison required')
    if (file_hash(root/'study.json') != report['study_sha256'] or
            file_hash(root/'evaluation-protocol.json') != report['evaluation_protocol_sha256']):
        raise ValueError('verified study identity differs')
    study = json.loads((root/'study.json').read_text())
    rows, versions = [], {'initial': {}, 'final': {}}
    for role in ('learning', 'selection'):
        for policy in versions:
            stage = root/(policy+'-'+role)
            audit = report['audits'][stage.name]
            if Path(audit['root']).resolve() != stage.resolve() or not audit['passed']:
                raise ValueError('verified stage identity differs')
            for path, digest in audit['files'].items():
                if file_hash(path) != digest: raise ValueError('verified prediction files changed')
            c = json.loads((stage/'config.json').read_text())
            current = configured_rule_rows(c)
            predictions = [json.loads(s) for s in (stage/'run/records.jsonl').read_text().splitlines()]
            if len(predictions) != len(current) or {r['id'] for r in current} != {r['id'] for r in predictions}:
                raise ValueError('population changed')
            if policy == 'initial': rows.extend(current)
            versions[policy].update({r['id']: r for r in predictions})
    by_id = {r['id']: r for r in rows}
    cells = {}
    for key, cell in study['evaluation_cells'].items():
        selected = [by_id[uid] for uid in cell['request_ids']]
        cells[key] = {policy: diagnose(selected, [versions[policy][r['id']] for r in selected])
                      for policy in versions}
    return dict(cells=cells, verified_comparison_sha256=file_hash(verified_comparison),
        study_sha256=report['study_sha256'], scope='Post-hoc descriptive diagnosis of all prespecified cells '
        'and arms. No new inference, causal claim, changed training reward or checkpoint selection.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('root','verified-comparison','output'): parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args(); result=analyze(args.root,args.verified_comparison); atomic_json(args.output,result)
    print(json.dumps({cell: data['final']['arms']['structured'] for cell,data in result['cells'].items()},indent=2))
