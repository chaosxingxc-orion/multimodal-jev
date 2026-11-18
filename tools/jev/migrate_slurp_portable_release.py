"""Publish a new self-contained revision; never mutate the input release."""
import argparse
import json
from pathlib import Path
from multimodal_jev.local_config import load_config
from multimodal_jev.slurp_portable_release import migrate
from multimodal_jev.baseline import file_hash
from tools.jev.verify_slurp_official_intent import verify


def run(c):
    output=Path(c['output']);result=migrate(c['source'],c['source_manifest_sha256'],output)
    independent=verify(output,file_hash(output/'MANIFEST.json'))
    report=dict(schema='slurp-portable-migration-audit-v1',source_manifest_sha256=c['source_manifest_sha256'],
        manifest_sha256=file_hash(output/'MANIFEST.json'),counts=result['counts'],
        unchanged_payload_sha256=result['migration']['unchanged_payload_sha256'],independent=independent,
        scope='Identical admitted population, task and saved audio receipts; self-contained admission evidence; audio blobs not copied')
    report_path=Path(c['report']);report_path.parent.mkdir(parents=True,exist_ok=True)
    with report_path.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,_=load_config(a.config,a.local);print(json.dumps(run(c),indent=2))
