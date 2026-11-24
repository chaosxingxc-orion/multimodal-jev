"""Freeze label-free independent HF outputs from an audited finite replay."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_execution import content_key
from multimodal_jev.local_checkpoint import commit_json


def read(path):
    return json.loads(path.read_bytes())


def collect(root, manifest_path, output, candidate_modes):
    ops = root/'ops'
    if read(ops/'terminal.json') != {'exit_code': 0, 'status': 'declared_stages_finished'}:
        raise ValueError('finite replay did not finish successfully')
    manifest = read(manifest_path)
    plan = manifest['plan']
    digest = content_key(plan)
    preparation = read(ops/'block-plan-collected.json')
    if not digest == manifest['sha256'] == preparation['plan_sha256']:
        raise ValueError('finite replay plan differs from its collected receipt')
    expected = {(b['dataset'], b['arm'], b['key']): b for b in plan['blocks']}
    counts = defaultdict(int)
    modes = dict.fromkeys(plan['inputs'], 'bounded_batch') | candidate_modes
    if modes.keys() != plan['inputs'].keys():
        raise ValueError('unknown inference mode dataset')
    fail = defaultdict(int)
    times = defaultdict(lambda: dict(pairs=0, seconds=0.0, reference_seconds=0.0))
    blocks = {}
    source_pins = {}
    worker_paths = sorted(ops.glob('block-worker-*-collected.json'))
    for path in worker_paths:
        receipt = read(path)
        if (receipt['plan_sha256'] != digest or receipt['labels_used'] is not False
                or receipt['npu_executed'] is not True):
            raise ValueError('invalid independent NPU replay receipt')
        source_pins[path.name] = file_hash(path)
        for row in receipt['diagnostic']:
            dataset, arm, key = row['dataset'], row['arm'], row['block']
            block = expected.pop((dataset, arm, key))
            if row['policy_id'] != plan['policies'][arm] or not row['batches']:
                raise ValueError('wrong independent HF policy or empty block')
            mini = []
            for batch in row['batches']:
                examples = []
                if batch['pairs'] != len(batch['neighbors']) or not batch['examples']:
                    raise ValueError('incomplete batch geometry or comparisons')
                elapsed = batch['batch_seconds']
                if modes[dataset] not in ('bounded_batch', 'hf_singleton'):
                    elapsed = batch['examples'][0]['variants'][modes[dataset]]['seconds']
                timing = times[dataset]
                timing['pairs'] += batch['pairs']
                timing['seconds'] += elapsed
                for x in batch['examples']:
                    pos, uid = x['index'], x['id']
                    if not 0 <= pos < batch['pairs'] or batch['neighbors'][pos]['id'] != uid:
                        raise ValueError('reference membership differs from its original batch')
                    recording, candidate = uid.rsplit(':', 1)
                    if recording not in block['ids'] or not 0 <= int(candidate) < block['widths'][block['ids'].index(recording)]:
                        raise ValueError('reference pair is outside the frozen block')
                    z, p = x['singleton_logit'], x['singleton_probability']
                    if not math.isfinite(z) or not 0 <= p <= 1:
                        raise ValueError('invalid independent HF probability')
                    if modes[dataset] != 'hf_singleton':
                        observed = (x['variants'][modes[dataset]] if modes[dataset] != 'bounded_batch' else
                                    dict(logit=x['batch_logit'], probability=x['batch_probability']))
                        fail[dataset] += int(observed['logit'] != z or observed['probability'] != p)
                    counts[dataset] += 1
                    timing['reference_seconds'] += x['singleton_seconds']
                    examples.append(dict(index=pos, id=uid, singleton_logit=z, singleton_probability=p))
                mini.append(dict(neighbors=batch['neighbors'], examples=examples))
            blocks[key] = dict(block=key, dataset=dataset, arm=arm, policy_id=row['policy_id'], batches=mini)
    if expected or counts.keys() != plan['inputs'].keys() or len(worker_paths) < 2:
        raise ValueError('incomplete finite HF reference coverage')
    modes.update({dataset: 'hf_singleton' for dataset, count in fail.items() if count})
    reference = dict(schema='jev-single-card-hf-panel-v1', policies=plan['policies'], inputs=plan['inputs'],
                     replay_plan_sha256=digest, receipt_sha256=source_pins,
                     blocks=[blocks[b['key']] for b in plan['blocks']])
    commit_json(output, reference)
    return dict(reference=dict(path=str(output), sha256=file_hash(output)), modes=modes,
                differences=dict(fail), counts=dict(counts), timing=dict(times))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--candidate-modes', type=json.loads, default={})
    args = parser.parse_args()
    print(json.dumps(collect(args.root, args.manifest, args.output, args.candidate_modes), ensure_ascii=False))
