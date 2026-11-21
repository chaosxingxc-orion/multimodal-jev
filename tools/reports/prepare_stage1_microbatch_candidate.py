"""Prepare a bounded Stage1 microbatch candidate using shared schedule/runtime."""
import argparse,json
from pathlib import Path
from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.baseline import file_hash
from tools.reports.prepare_context_replica_schedule import prepare as schedule
from tools.reports.context_instruction_operations import prepare as package

def prepare(c,provenance):
    p=dict(c['package']);design=json.loads(Path(p['design']).read_text())
    schedule(dict(output=c['schedule_output'],release=p['release'],release_sha256=p['release_sha256'],parent_schedule=p['parent_schedule'],parent_manifest_sha256=p['parent_schedule_sha256'],parent_release_sha256=p['parent_release_sha256'],batch_config=design['stage1']['batch_config']),provenance)
    p.update(schedule=c['schedule_output'],schedule_sha256=file_hash(Path(c['schedule_output'])/'MANIFEST.json'))
    result=package(p,provenance)
    commit_json(c['result'],dict(result=result,package=p))
    return result
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--local',required=True);x=a.parse_args();c,p=load_config(x.config,x.local);print(json.dumps(prepare(c,p)))
