#!/usr/bin/env python3
"""Version an authenticated full SLURP schedule with class temporal coverage.

Only optimizer membership/order changes. Original request fields, weights and
population remain fixed. Processor-coordinate admission is explicitly pending.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.simple_rl_schedule import SimpleRLSchedule
from multimodal_jev.slurp_official_training import load
from multimodal_jev.slurp_training import content_hash
from multimodal_jev.span_class_spread import class_spread_indices


def prepare(parent, parent_sha256, output):
    parent, output = Path(parent), Path(output)
    if output.exists():
        raise FileExistsError('immutable schedule already exists')
    if file_hash(parent/'MANIFEST.json') != parent_sha256:
        raise ValueError('authenticated parent required')
    manifest = json.loads((parent/'MANIFEST.json').read_text())
    rows, _, _ = load(manifest['release'], manifest['manifest_sha256'], 'learning', purpose='optimize')
    original = SimpleRLSchedule(parent, manifest_sha256=parent_sha256,
                                release_sha256=manifest['manifest_sha256'])
    original.validate_population(rows)
    labels = {r['id']: r['intent'] for r in rows}
    output.mkdir(parents=True)
    reports = []
    diagnostics = []
    for epoch in range(2):
        wire = json.loads((parent/f'epoch-{epoch}.json').read_text())
        real = [slot for batch in wire['batches'] for slot in batch if not slot['padding']]
        indices = class_spread_indices([labels[s['id']] for s in real], seed=317, epoch=epoch)
        slots = [dict(real[i]) for i in indices]
        if sorted(slots, key=lambda s: s['id']) != sorted(real, key=lambda s: s['id']):
            raise ValueError('real request or weight changed')
        real_count = len(slots)
        while len(slots) % 128:
            slots.append(dict(slots[len(slots)-real_count], padding=True, weight=0.))
        batches = [slots[i:i+128] for i in range(0, len(slots), 128)]
        report = dict(wire['report'])
        report.update(schema='slurp-class-spread-epoch-v1', ordering='class_normalized_progress_v1',
            parent_manifest_sha256=parent_sha256, parent_epoch_sha256=file_hash(parent/f'epoch-{epoch}.json'),
            batches_sha256=content_hash(batches), class_interleaved=False, class_temporally_spread=True)
        report.pop('schedule_sha256')
        report['schedule_sha256'] = content_hash(report)
        path = output/f'epoch-{epoch}.json'
        path.write_bytes(canonical(dict(report=report, batches=batches)))
        reports.append(dict(report, file_sha256=file_hash(path)))
        windows = []
        for start, end in [(0,32),(100,132),(200,232),(285,317),(309,317),(316,317)]:
            counts = Counter(labels[s['id']] for b in batches[start:end] for s in b if not s['padding'])
            windows.append(dict(start=start, end_exclusive=end, classes=len(counts), counts=dict(counts)))
        diagnostics.append(dict(epoch=epoch, windows=windows))
    files = [Path(__file__), Path(__file__).resolve().parents[2]/'src/multimodal_jev/span_class_spread.py']
    result = dict(manifest)
    result.update(schema='slurp-class-spread-epochs-v1', epochs=reports,
        ordering='class_normalized_progress_v1', parent_manifest_sha256=parent_sha256,
        NPU_preflight_passed=False, training_dispatched=False, training_dispatch_allowed=False,
        pins={str(p.resolve()):file_hash(p) for p in files},
        status='new_order_population_verified_processor_coordinates_and_runtime_pending')
    (output/'MANIFEST.json').write_bytes(canonical(result))
    proposed = SimpleRLSchedule(output, manifest_sha256=file_hash(output/'MANIFEST.json'),
                               release_sha256=manifest['manifest_sha256'])
    admission = proposed.validate_population(rows)
    if proposed.identities != original.identities:
        raise ValueError('proposed full population differs from parent')
    receipt = dict(passed=True, **{k:v for k,v in admission.items() if k!='passed'},
        same_population_source_groups_and_weights=True, diagnostics=diagnostics,
        parent_manifest_sha256=parent_sha256, files=proposed.files,
        training_dispatch_allowed=False,
        scope='New immutable full schedule and independent existing loader/weight checks. '
              'Different global memberships; no claim of optimizer-step equivalence or new model performance. '
              'Processor-coordinate preflight required before training.')
    (output/'population-verified.json').write_bytes(canonical(receipt))
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--parent-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = prepare(args.parent, args.parent_sha256, args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('diagnostics','files')}, indent=2))
