"""Run two finite NPU measurement processes, retaining every failure/log."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def verify(package):
    package=Path(package)
    files=json.loads((package/'FILES.json').read_text())
    actual={str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in package.rglob('*') if p.is_file() and p!=package/'FILES.json'}
    if any(p.is_symlink() for p in package.rglob('*')) or actual!=files:
        raise ValueError('sealed probe package changed')
    return hashlib.sha256((package/'FILES.json').read_bytes()).hexdigest()


def run(package, output):
    from multimodal_jev.local_runtime_storage import require_training_storage
    from multimodal_jev.local_checkpoint import commit_json
    from tools.reports.context_batch_measurement_candidate import compare
    package,output=Path(package),Path(output)
    require_training_storage(output);output.mkdir(exist_ok=False)
    try:
        identity=verify(package);commit_json(output/'source.json',dict(files_sha256=identity))
        for mode in ('serial','batched'):
            argv=[sys.executable,'-B',str(package/'tools/reports/context_batch_measurement_candidate.py'),
                  '--config',str(package/'config.json'),'--output',str(output/mode),'--mode',mode]
            commit_json(output/(mode+'-command.json'),dict(argv=argv))
            with (output/(mode+'.log')).open('x') as log:
                result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
            commit_json(output/(mode+'-exit.json'),dict(exit_code=result.returncode))
            if result.returncode:raise RuntimeError(mode+' probe failed; retained log and exit')
        a,b=[json.loads((output/mode/'measurement.json').read_text()) for mode in ('serial','batched')]
        result=compare(a,b);commit_json(output/'comparison.json',result)
        commit_json(output/'terminal.json',dict(exit_code=0));return result
    except BaseException as exc:
        commit_json(output/'terminal.json',dict(exit_code=1,error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.package,a.output)
