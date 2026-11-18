"""Fixed offline padding comparison using actual baseline processor shapes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_length_buckets import plan
from multimodal_jev.local_config import load_config
from multimodal_jev.span_length_groups import length_group_indices


def summarize(length_batches, settings, *, microbatch=8, world_size=2):
    totals = {mode: dict(padded_tokens=0, padded_frames=0, slowest_rank_cost_proxy=0)
              for mode in ('baseline', 'candidate')}
    useful = dict(tokens=0, frames=0); selected = Counter(); steps = []
    for lengths in length_batches:
        ranks, report = plan(lengths, microbatch=microbatch, world_size=world_size, **settings)
        selected[report['selected']] += 1
        for mode in totals:
            for key in totals[mode]: totals[mode][key] += report[mode][key]
        useful['tokens'] += sum(t for t, f in lengths)
        useful['frames'] += sum(f for t, f in lengths)
        steps.append(report)
    if not steps: raise ValueError('nonempty actual length batches required')
    for mode, values in totals.items():
        values['token_padding_fraction'] = 1-useful['tokens']/values['padded_tokens']
        values['frame_padding_fraction'] = 1-useful['frames']/values['padded_frames']
    reduction = {k: 1-totals['candidate'][k]/totals['baseline'][k]
                 for k in ('padded_tokens', 'padded_frames', 'slowest_rank_cost_proxy')}
    return dict(updates=len(steps), selected=dict(selected), useful=useful,
        totals=totals, reduction_fraction=reduction, steps=steps,
        measured_speedup=None,
        scope='Offline fixed-batch padding and allocation proxy; not NPU runtime or FLOPs.')


def analyze(c):
    for path, digest in c['pins'].items():
        if file_hash(path) != digest: raise ValueError('analysis dependency changed: '+path)
    base = Path(c['operations_root']); transport = json.loads((base/'transport.json').read_text())
    def read(name):
        path = base/'collected'/name
        if path.is_symlink() or file_hash(path) != transport['files'][name]:
            raise ValueError('collected input differs: '+name)
        return json.loads(path.read_text())
    result = read('run/result.json')
    if result['passed'] is not True or result['start_update'] != 0 or result['stop_update'] != 100:
        raise ValueError('completed100 baseline required')
    length_batches = []; source_files = {}
    for i in range(100):
        names = [f'run/batch-{i:04d}.json', f'run/rollout-{i:04d}.json']
        batch, rollout = [read(n) for n in names]
        source_files.update({n: transport['files'][n] for n in names})
        records = batch['records']; queries = rollout['queries']
        if (len(records) != 128 or len(queries) != 128 or rollout['update'] != i
                or [r['request_id'] for r in records] != [q['request_id'] for q in queries]
                or len({q['request_id'] for q in queries}) != 128):
            raise ValueError('actual full-batch request identities differ')
        lengths = []
        for q in queries:
            r = q['behavior_record']
            # Engine request IDs hash tensor/policy identity; outer IDs identify
            # schedule slots and intentionally use a different namespace.
            identity_id = hashlib.sha256(json.dumps(r['request_identity'],
                sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            if (r['request_id'] != identity_id or r['policy_id'] != batch['policy_id']
                    or r['request_identity']['policy_id'] != r['policy_id']):
                raise ValueError('behavior request or policy mismatch')
            shapes = r['request_identity']['inputs']
            lengths.append((shapes['input_ids']['shape'][-1], shapes['input_features']['shape'][-1]))
        baseline = length_group_indices(lengths, microbatch=8, world_size=2, order='length_grouped')
        if baseline != batch['rank_indices']:
            raise ValueError('actual baseline rank allocation differs')
        length_batches.append(lengths)
    report = summarize(length_batches, c['length_buckets'])
    return dict(report, schema='jev-bucket-padding-analysis-v1', settings=c['length_buckets'],
        queries=12800, source_files=source_files, transport_sha256=file_hash(base/'transport.json'),
        training_or_promotion_allowed=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--local', required=True)
    args = p.parse_args(); c, provenance = load_config(args.config, args.local)
    report = analyze(c); report['config_provenance'] = provenance
    output = Path(c['output']); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f: json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps({k: report[k] for k in ('updates', 'queries', 'selected', 'totals', 'reduction_fraction', 'scope')}, indent=2))
