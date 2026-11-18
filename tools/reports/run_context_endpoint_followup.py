"""Local-only endpoint reports after the original experiment owner finishes.

Never holds the predecessor lock while auditing, creates a predecessor lock,
launches remote work, selects checkpoints or authorizes complex training.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import traceback


def read(path):return json.loads(Path(path).read_text())


def verify_pins(pins):
    for name,digest in pins.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise ValueError('analysis dependency changed: '+name)


def ready(directory):
    root=Path(directory);lock=root/'owner.lock'
    if not lock.is_file():return False
    # r+ is deliberately not used: missing owner files must never be created.
    with lock.open('r') as owner:
        try:fcntl.flock(owner,fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError:return False
        phase=read(root/'status.json')['phase']
        if phase!='complete_training_and_selection_curve':
            raise ValueError('original experiment ended without complete curve: '+phase)
    return True


def analyze(entry,release):
    from tools.reports.context_probability_endpoint_uncertainty import run
    return run(entry['directory'],entry['algorithm'],release)


def finish_one(entry,release,pins,*,compute=analyze):
    output=Path(entry['output'])
    if output.exists():raise ValueError('existing report; preserve without overwrite')
    verify_pins(pins)
    if not ready(entry['directory']):return False
    result=compute(entry,release)
    verify_pins(pins)
    if result.get('complex_training_admitted') is not False:
        raise ValueError('report must not authorize complex training')
    with output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return True


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',required=True)
    args=parser.parse_args();spec=read(args.spec);root=Path(spec['output_directory'])
    root.mkdir(exist_ok=True)
    with (root/'owner.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'status.json').exists():raise ValueError('existing followup; preserve original state')
        def status(phase,**extra):
            p=root/'status.json';temp=p.with_suffix('.tmp')
            temp.write_text(json.dumps(dict(phase=phase,pid=os.getpid(),unix=time.time(),**extra),indent=2)+'\n');temp.replace(p)
        completed=[]
        try:
            verify_pins(spec['pins'])
            for entry in spec['experiments']:
                while True:
                    status('waiting_original_experiment',algorithm=entry['algorithm'],completed=completed)
                    if ready(entry['directory']):break
                    time.sleep(60)
                status('auditing_complete_curve_and_endpoint',algorithm=entry['algorithm'],completed=completed)
                if not finish_one(entry,spec['release'],spec['pins']):
                    raise ValueError('predecessor ownership changed')
                completed.append(entry['output'])
            status('complete_local_endpoint_reports',completed=completed,complex_training_admitted=False)
        except BaseException as exc:
            status('stopped_preserve_evidence',error=repr(exc),traceback=traceback.format_exc(),completed=completed)
            raise


if __name__=='__main__':main()
