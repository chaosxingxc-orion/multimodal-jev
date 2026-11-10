#!/usr/bin/env python3
"""Estimate one complete registered population from an accepted TRAIN78 run."""
import argparse
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical
from multimodal_jev.sfc_jev_shard_runtime import read,check_probe
from multimodal_jev.sfc_jev_shards import verify_plan,population
from multimodal_jev.sfc_runtime_cost import project
from verify_sfc_jev_probe import accept


def estimate(probe,plan_path,plan_hash):
    probe=Path(probe);plan_path=Path(plan_path)
    if file_hash(plan_path)!=plan_hash:raise ValueError('registered complete plan changed')
    plan=read(plan_path);verify_plan(plan)
    cfg=read(probe/'config.json');launch=read(probe/'launch.json')
    check_probe(dict(cfg,jev_probe=str(probe),jev_probe_container_id=launch['container_id'],
        jev_probe_acceptance_sha256=file_hash(probe/'acceptance.json'),image=launch['image']))
    observed,identity,predictions=accept(probe)  # Exact successful Docker + all original outputs.
    if (Path(identity['release']).resolve()!=Path(plan['release']).resolve()
            or identity['manifest_sha256']!=plan['manifest_sha256']):
        raise ValueError('same accepted data release required')
    result=read(probe/'run/result.json')
    wanted={uid for shard in plan['shards'] for uid in shard['ordered_bundle_ids']}
    rows=[r for r in population(plan['release'],plan['split']) if r['id'] in wanted]
    report=project(observed,predictions,result['elapsed_seconds'],rows,plan)
    report.update(schema='sfc-jev-parent-runtime-estimate-v1',probe=str(probe.resolve()),
        probe_container_id=launch['container_id'],image=launch['image'],
        probe_result_sha256=file_hash(probe/'run/result.json'),
        probe_predictions_sha256=file_hash(probe/'run/predictions.jsonl'),
        probe_acceptance_sha256=file_hash(probe/'acceptance.json'),
        plan=str(plan_path.resolve()),plan_sha256=plan_hash,split=plan['split'],partition=plan['partition'],
        versions=cfg['versions'],audio_model=cfg['audio_model'],max_new_tokens=cfg['max_new_tokens'],
        physical_devices=[6,7],targets_read_by_estimator=False,
        source_files={n:file_hash(Path(__file__).parents[2]/n) for n in
            ('tools/jev/estimate_sfc_jev_cost.py','src/multimodal_jev/sfc_runtime_cost.py')})
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('probe','plan','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--plan-hash',required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve prior cost evidence')
    report=estimate(a.probe,a.plan,a.plan_hash)
    with a.output.open('xb') as handle:handle.write(canonical(report))
    print(a.output)
