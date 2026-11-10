#!/usr/bin/env python3
"""Accept complete official TEST inference and compare frozen prompts strictly."""
import argparse
import inspect
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.omni_test_contract import select_test, GROUPS
from multimodal_jev.six_view_score import score_six_views
from select_omni_prompt import paired


def load(root, protocol_path, key):
    protocol = json.loads(protocol_path.read_text())
    identity = json.loads((root/'identity.json').read_text())
    cfg = identity['config']
    result = json.loads((root/'result.json').read_text())
    launch = json.loads((root.parent/'launch.json').read_text())
    terminal = json.loads((root.parent/'docker-terminal.json').read_text())
    if (terminal['container_id'] != launch['container_id']
            or terminal['state']['Status'] != 'exited' or terminal['state']['ExitCode'] != 0
            or launch['image'] != protocol['image'] or launch['physical_devices'] != [4,5]
            or json.loads((root/'terminal.json').read_text())['exit_code'] != 0
            or json.loads((root.parent/'evaluation-terminal.json').read_text())['exit_code'] != 0
            or result['status'] != 'passed' or result['weight_updates'] != 0
            or result['model_files_unchanged'] is not True or result['audio_condition'] != 'actual'):
        raise ValueError('complete successful frozen TEST container and application required')
    if (file_hash(protocol_path) != cfg['test_protocol_sha256']
            or identity['test_protocol_sha256'] != cfg['test_protocol_sha256']
            or identity['instruction_sha256'] != protocol['prompts'][key]
            or identity['instruction_sha256'] != file_hash(cfg['instruction'])
            or identity['requests_sha256'] != protocol['requests_sha256']
            or identity['asset_audit_sha256'] != protocol['asset_audit_sha256']
            or any(identity[k] != v for k,v in protocol['actor_identity'].items())
            or identity['population_selector_sha256'] != file_hash(inspect.getfile(select_test))
            or file_hash(root/'predictions.jsonl') != result['predictions_sha256']):
        raise ValueError('TEST input, runtime or output identity changed')
    rows = select_test(cfg,[json.loads(line) for line in Path(cfg['requests']).open()])
    predictions = [json.loads(line) for line in (root/'predictions.jsonl').open()]
    indexed = {p['example_id']:p for p in predictions}
    ids = json.loads((root/'selection.json').read_text())['ids']
    if (ids != [r['id'] for r in rows] or len(predictions) != GROUPS*6
            or len(indexed) != GROUPS*6 or set(indexed) != set(ids)
            or result['requests'] != GROUPS*6 or result['predictions'] != GROUPS*6
            or any(p['replicate'] != 0 or p['split'] != 'test' or p['audio_condition'] != 'actual'
                   for p in predictions)):
        raise ValueError('unique full TEST predictions required')
    return identity, rows, indexed, result


def compare(root):
    protocol_path = root/'test-protocol.json'
    protocol = json.loads(protocol_path.read_text())
    parent, rows, predictions, _ = load(root/'parent/run',protocol_path,'parent')
    target_path = Path(parent['config']['requests']).with_name('test.targets.jsonl')
    if file_hash(target_path) != protocol['targets_sha256']:
        raise ValueError('official TEST targets changed')
    targets = [json.loads(line) for line in target_path.open()]
    reports = []
    for key in protocol['prompts']:
        identity, candidate_rows, indexed, result = load(root/key/'run',protocol_path,key)
        if rows != candidate_rows:
            raise ValueError('paired TEST population differs')
        scored = score_six_views(rows,targets,list(indexed.values()))
        if scored['source_groups'] != GROUPS or len(scored['replicates']) != 1:
            raise ValueError('complete single TEST replicate required')
        diagnostic = paired(rows,targets,predictions,indexed)
        diagnostic['scope'] = 'official TEST paired comparison; views correlated, not pooled'
        reports.append(dict(id=key,instruction_sha256=identity['instruction_sha256'],
            metrics=scored['replicates'][0],paired_to_parent=diagnostic,
            length_limited=result['length_limited'],result_sha256=file_hash(root/key/'run/result.json')))
    return dict(schema='jev-omni-official-test-comparison-v1',protocol_sha256=file_hash(protocol_path),
        source_groups=GROUPS,requests_per_prompt=GROUPS*6,comparisons=reports,weight_updates=0,
        targets_sha256=file_hash(target_path),test_feedback_used_for_optimization=False,
        scope='Full official FSC TEST native and five Jev views, frozen audio-direct; not weight-trained or cross-task evidence')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=compare(a.root)
    with a.output.open('xb') as handle: handle.write(canonical(report))
    print(json.dumps(report,indent=2))
