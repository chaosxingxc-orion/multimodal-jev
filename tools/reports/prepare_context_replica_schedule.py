"""Prepare a source-bound candidate schedule through portable local macros."""
import argparse
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_replica_schedule import rebatch
from multimodal_jev.context_topology_batch_plan import derive
from multimodal_jev.context_training_schedule import load_schedule
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from multimodal_jev.slurp_portable_release import load
from multimodal_jev.slurp_training import content_hash


def prepare(c, provenance):
    output=Path(c['output'])
    if output.exists():raise FileExistsError('immutable schedule already exists')
    rows,_,_=load(c['release'],c['release_sha256'],'learning',purpose='optimize')
    population={r['id']:r for r in rows}
    parent,_=load_schedule(c['parent_schedule'],c['parent_manifest_sha256'],
        release_sha256=c['parent_release_sha256'],population=population)
    batches=rebatch(parent,list(population),c['batch_config'])
    check_paths({'schedule':output});output.mkdir(parents=True)
    commit_json(output/'batches.json',batches)
    result=dict(schema='jev-replica-source-schedule-v1',
        release_sha256=c['release_sha256'],parent_manifest_sha256=c['parent_manifest_sha256'],
        parent_release_sha256=c['parent_release_sha256'],batch_plan=derive(c['batch_config']),
        real_population_sha256=content_hash(sorted(population)),
        files={'batches.json':file_hash(output/'batches.json')},
        total_updates=len(batches),total_real_exposures=sum(b['real_queries'] for b in batches),
        runtime_rank_assignment='length_group_actual_encoded_inputs_within_macro',
        training_dispatch_allowed=False,config_provenance=provenance)
    commit_json(output/'MANIFEST.json',result)
    return result


if __name__=='__main__':
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    a=parser.parse_args();c,p=load_config(a.config,a.local)
    print(json.dumps(prepare(c,p),indent=2))
