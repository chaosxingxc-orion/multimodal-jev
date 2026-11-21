"""Build and independently re-read one sealed native input population on CPU."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json
from tools.reports.prepare_native_seal_cache import run

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--local', required=True)
    p.add_argument('--verify', action='store_true')
    a = p.parse_args()
    c, meta = load_config(a.config, a.local)
    proof = Path(c['output'])/'VERIFIED.json'
    if a.verify:
        commit_json(proof, run(c, meta, verify=True))
    else:
        run(c, meta)
        subprocess.run([sys.executable, '-B', '-m', 'tools.reports.prepare_native_seal_cache_job',
                        '--config', a.config, '--local', a.local, '--verify'], check=True)
        print(json.dumps(json.loads(proof.read_text())))
