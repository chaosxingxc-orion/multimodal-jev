#!/usr/bin/env python3
"""Validate complete frozen inference before calling the official SLURP metrics."""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.slurp_score import load_official_metrics, score_native


def score_run(run, targets, repository, output):
    if output.exists(): raise FileExistsError('preserve scored result')
    identity = json.loads((run/'identity.json').read_text()); cfg = identity['config']
    result = json.loads((run/'result.json').read_text())
    if (json.loads((run/'terminal.json').read_text())['exit_code'] != 0
            or result['status'] != 'passed' or result['weight_updates'] != 0
            or cfg.get('task_format') != 'slurp-native'):
        raise ValueError('completed frozen native SLURP inference required')
    if cfg['sampling'] != {'temperature': 0, 'n': 1}: raise ValueError('single greedy evaluation required')
    if (cfg.get('route') != 'audio-direct' or result.get('route') != 'audio-direct'
            or cfg.get('parameters_frozen') is not True or cfg.get('adapter') is not None
            or result.get('model_files_unchanged') is not True):
        raise ValueError('frozen end-to-end audio inference required')
    condition = cfg.get('audio_condition', 'actual')
    if (condition not in ('actual', 'zeros') or result.get('audio_condition') != condition
            or (condition == 'zeros' and cfg['split'] != 'train')):
        raise ValueError('consistent actual audio or TRAIN-only zero-audio diagnostic required')
    request_path = Path(cfg['requests']); release = request_path.parent
    manifest = json.loads((release/'MANIFEST.json').read_text())
    split = cfg['split']; pins = manifest['splits'][split]
    if (request_path.name != split+'.requests.jsonl' or
            targets.resolve() != (release/(split+'.targets.jsonl')).resolve()):
        raise ValueError('paired official split files required')
    if (file_hash(request_path) != pins['requests_sha256']
            or pins['requests_sha256'] != identity['requests_sha256']
            or file_hash(targets) != pins['targets_sha256']):
        raise ValueError('native release input changed')
    taxonomy = json.loads((release/'taxonomy.json').read_text())
    if file_hash(release/'taxonomy.json') != manifest['identity']['taxonomy_sha256']:
        raise ValueError('release taxonomy changed')
    if identity.get('taxonomy_sha256') != manifest['identity']['taxonomy_sha256']:
        raise ValueError('inference taxonomy differs from native release')
    requests = list(map(json.loads, request_path.read_text().splitlines()))
    if len(requests) != pins['recordings']: raise ValueError('official recording count changed')
    if cfg.get('source_groups') is not None:
        groups = set(cfg['source_groups'])
        if len(groups) != len(cfg['source_groups']) or not groups <= {r['source_group'] for r in requests}:
            raise ValueError('invalid configured group selection')
        requests = [r for r in requests if r['source_group'] in groups]
    selected = json.loads((run/'selection.json').read_text())['ids']
    if selected != [r['id'] for r in requests] or any(r['split'] != split for r in requests):
        raise ValueError('selection differs from configured full recording groups')
    if file_hash(run/'predictions.jsonl') != result['predictions_sha256']:
        raise ValueError('prediction hashes changed')
    predictions = list(map(json.loads, (run/'predictions.jsonl').read_text().splitlines()))
    if (type(result.get('requests')) is not int or result['requests'] != len(requests)
            or type(result.get('predictions')) is not int or result['predictions'] != len(predictions)):
        raise ValueError('inference population counts differ')
    if any(p.get('audio_condition') != condition for p in predictions):
        raise ValueError('prediction audio condition differs')
    report, normalized = score_native(requests, list(map(json.loads, targets.read_text().splitlines())),
                                      predictions, taxonomy, load_official_metrics(repository))
    report.update(split=split, purpose=cfg['purpose'], full_official_split=cfg.get('source_groups') is None,
                  release_id=manifest['release_id'], run=str(run), result_sha256=file_hash(run/'result.json'),
                  targets_sha256=file_hash(targets), scorer_source_manifest_sha256=file_hash(repository/'FETCH.json'),
                  identity_sha256=file_hash(run/'identity.json'), route=cfg['route'], audio_condition=condition,
                  audio_task_result=condition == 'actual', models=identity['models'],
                  instruction_sha256=identity['instruction_sha256'],
                  inference_runner_sha256=identity['runner_sha256'])
    output.mkdir(parents=True, exist_ok=False)
    (output/'official-predictions.jsonl').write_bytes(b''.join(map(canonical, normalized)))
    (output/'result.json').write_bytes(canonical(report))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('run','targets','repository','output'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(score_run(args.run,args.targets,args.repository,args.output),indent=2))
