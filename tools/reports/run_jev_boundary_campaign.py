"""Run and collect a paired native diagnostic with CPU and NPU leases separated."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.native_seal_operations import run
from tools.reports.prepare_jev_boundary_runtime import reuse_prep
from tools.reports.score_jev_boundary_panel import score


def campaign(c):
    root=Path(c['output'])
    receipt=json.loads((root/'receipt.json').read_bytes())
    try:
        if file_hash(Path(c['seal'])/'MANIFEST.json')!=receipt['seal_sha256']:
            raise ValueError('paired seal changed')
        if 'reuse_prep' in receipt:
            if c.get('reuse_prep') != receipt['reuse_prep']:
                raise ValueError('CPU cache binding changed')
            reuse_prep(c, json.loads((Path(c['seal'])/'MANIFEST.json').read_bytes()), receipt['policy'])
        phases = ('hf',) if 'reuse_prep' in receipt else ('prep', 'hf')
        for phase in phases:
            base=root/phase
            for name,field in [('local.json','local_sha256'),('entry.tar.gz','archive_sha256')]:
                if file_hash(base/name)!=receipt['packages'][phase][field]:raise ValueError('job changed')
            cfg=json.loads((base/'local.json').read_bytes())['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']
            if run(cfg) != 0:
                raise ValueError('diagnostic phase failed: ' + phase)
        commit_json(root/'paired-score.json',score(c))
        commit_json(root/'terminal.json',dict(exit_code=0,score_sha256=file_hash(root/'paired-score.json')))
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc)))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--detach', action='store_true')
    args = p.parse_args()
    cfg, _ = load_config(args.config, args.local)
    if not args.detach:
        campaign(cfg)
        return
    root = Path(cfg['output'])
    if not (root / 'receipt.json').is_file():
        raise ValueError('prepare the diagnostic package first')
    driver = root / 'driver'
    check_paths({'diagnostic driver': driver})
    driver.mkdir()
    argv = [sys.executable, '-B', '-m', __spec__.name, '--config', str(Path(args.config).resolve()),
            '--local', str(Path(args.local).resolve())]
    with (driver / 'driver.log').open('x') as log:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    tick = Path(f'/proc/{child.pid}/stat').read_text().split(') ', 1)[1].split()[19]
    commit_json(driver / 'started.json', dict(pid=child.pid, process_start_ticks=tick, argv=argv))
    print(child.pid)


if __name__ == '__main__':
    main()
