"""Reconstruct frozen clean TEST truth after all inference roles are audited."""
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from tools.reports.evaluate_context_transfer_macro import read_bound
from tools.reports.score_slurp_context_test_pair import reconstruct_targets


def rows(path):
    with Path(path).open() as stream:return [json.loads(line) for line in stream]


def load(config,scoring,admitted):
    contract=read_bound(scoring['contract'])
    if (contract['schema']!='slurp-context-clean-test-scoring-v1'
            or contract['threshold']!=.5 or contract['optimization_allowed'] is not False):
        raise ValueError('frozen clean TEST scoring contract required')
    assets=contract['artifacts'];root=Path(config['inputs']['inputs'])
    for key,name in [('input_manifest','MANIFEST.json'),('definitions','definitions.json'),('input_rows','inputs.jsonl')]:
        if file_hash(root/name)!=assets[key]['sha256']:
            raise ValueError('TEST scoring inputs differ from inference')
    # All sources were pinned before inference; open labels only at this stage.
    for item in assets.values():
        if file_hash(item['path'])!=item['sha256']:
            raise ValueError('frozen TEST scoring source changed')
    manifest=read_bound(assets['input_manifest']);task=read_bound(assets['tasks'])['intent']
    definitions=read_bound(assets['definitions'])
    if (assets['tasks']['sha256']!=manifest['task_source_sha256']
            or definitions['candidates']!=[dict(id=c['id'],definition=c['description']) for c in task['schema']['candidates']]
            or [c['id'] for c in task['schema']['candidates']]!=admitted['candidate_ids']):
        raise ValueError('frozen TEST ontology differs from executed panel')
    for key in ('native_requests','native_targets','clean_metadata','excluded','population','reference_predictions','reference_result'):
        item=assets[key]
        if manifest['source_files'].get(item['path'])!=item['sha256']:
            raise ValueError('scoring source not bound to original input publication: '+key)
    inputs=rows(assets['input_rows']['path'])
    if len(inputs)!=12991 or inputs!=admitted['source_rows']:
        raise ValueError('complete executed clean TEST population required')
    truth=reconstruct_targets(rows(assets['native_requests']['path']),rows(assets['native_targets']['path']),
        rows(assets['clean_metadata']['path']),read_bound(assets['excluded']),read_bound(assets['population']),inputs,task)
    return inputs,truth
