#!/usr/bin/env python3
"""Freeze complete two-epoch instruction schedule from an authenticated release."""
import argparse
import json
from pathlib import Path
from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.slurp_official_training import load, example
from multimodal_jev.slurp_instruction_schedule import epoch_schedule


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--global-batch', type=int, default=128)
    p.add_argument('--task-mode', choices=['instruction_views', 'direct_final'], default='instruction_views')
    p.add_argument('--class-weighting', choices=['natural', 'sqrt_cap4'], default='natural')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('immutable schedule already exists')
    rows, task, manifest = load(a.release, a.manifest_sha256, 'learning', purpose='optimize')
    # Check every source target against every registered view without storing
    # duplicated audio features or multi-gigabyte repeated prompt definitions.
    by_id = {row['id']: row for row in rows}
    a.output.mkdir(parents=True)
    reports = []
    for epoch in range(2):
        batches, report = epoch_schedule(rows, seed=317, epoch=epoch, global_batch=a.global_batch, world_size=4,
                                        task_mode=a.task_mode, class_weighting=a.class_weighting)
        for batch in batches:
            for slot in batch:
                request, target, binding = example(by_id[slot['id']], task, slot['view'], slot['variant'])
                if [q.values[y] for q, y in zip(request.questions, target['targets'], strict=True)] != target['target_values']:
                    raise ValueError('semantic target mapping changed')
        path = a.output/f'epoch-{epoch}.json'
        path.write_bytes(canonical(dict(report=report, batches=batches)))
        reports.append(dict(**report, file_sha256=file_hash(path)))
    result = dict(schema='slurp-instruction-epochs-v1', release=str(a.release.resolve()),
        manifest_sha256=a.manifest_sha256, epochs=reports, global_batch=a.global_batch,
        world_size=4, microbatch=1, accumulation_per_rank=a.global_batch//4,
        unique_recordings=len(rows), total_nonpadding_recording_uses=len(rows)*2,
        train_source_groups=manifest['counts']['learning']['source_groups'],
        stage='instruction', objective='candidate-normalized-masked-CE',
        task_mode=a.task_mode, class_weighting=a.class_weighting,
        loss_reduction='real_request_mean' if a.task_mode == 'direct_final' else 'legacy_weight_mean',
        NPU_preflight_passed=False, training_dispatched=False,
        input_contract='audio plus input-only questions; targets outside backbone; fixed source-independent view/binding schedule',
        pins={str(Path(__file__)):file_hash(__file__),
              **{str(Path(__file__).resolve().parents[2]/'src/multimodal_jev'/f):
                 file_hash(Path(__file__).resolve().parents[2]/'src/multimodal_jev'/f)
                 for f in ('slurp_instruction_schedule.py','slurp_official_training.py','span_candidate_intervention.py')}})
    (a.output/'MANIFEST.json').write_bytes(canonical(result))
    print(json.dumps({k:v for k,v in result.items() if k not in ('pins','epochs')}, indent=2))


if __name__ == '__main__':
    main()
