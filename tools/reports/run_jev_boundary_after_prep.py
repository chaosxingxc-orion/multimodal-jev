"""Dispatch the prepared HF arm group only upon exact CPU prepare/verify success."""
import argparse
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from tools.reports.finalize_context_infonce import wait_exact
from tools.reports.native_seal_operations import run


def start(c):
    root=Path(c['output'])
    prep=root/'prep'
    receipt=json.loads((root/'receipt.json').read_bytes())
    for phase,expected in receipt['packages'].items():
        for name,key in [('local.json','local_sha256'),('entry.tar.gz','archive_sha256')]:
            if file_hash(root/phase/name)!=expected[key]: raise ValueError('prepared job changed')
    wait_exact(json.loads((prep/'run.started.json').read_bytes()),prep/'run/terminal.json')
    if json.loads((prep/'run/terminal.json').read_bytes())!=dict(exit_code=0,status='declared_stages_finished'):
        raise ValueError('input preparation failed')
    cfg=json.loads((root/'hf/local.json').read_bytes())['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']
    manifest=json.loads((Path(c['seal'])/'MANIFEST.json').read_bytes())
    for arm,data in manifest['datasets'].items():
        check=json.loads((prep/'run'/f'verify-{arm}-collected.json').read_bytes())
        if check['recordings']!=data['recordings'] or check['status']!='full_stable_prefix_cache_verified':
            raise ValueError('incomplete CPU check')
    run(cfg)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();start(load_config(a.config,a.local)[0])
