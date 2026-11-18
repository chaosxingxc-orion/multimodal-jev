"""Continue existing acceptance queues into the two frozen RL experiments.

Wait on local predecessor locks, never poll or restart their NPU containers.
Publish only through existing source/runtime validators. Failures stop the chain.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.jev.context_rl_evaluation_plan import sha


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def check_pins(pins):
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('queued baseline source changed: ' + path)


def wait_dependency(entry):
    with Path(entry['lock']).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if read(entry['status'])['phase'] != entry['success_phase']:
            raise ValueError('predecessor did not complete: ' + entry['status'])


def invoke(module, arguments, log):
    with Path(log).open('x') as stream:
        subprocess.run([sys.executable, '-m', 'tools.jev.' + module,
                        *map(str, arguments)], stdout=stream, stderr=subprocess.STDOUT, check=True)


def run(spec, output):
    check_pins(spec['pins'])
    if [x['algorithm'] for x in spec['runs']] != ['grpo_v1', 'rloo_v1']:
        raise ValueError('fixed GRPO then RLOO comparison required')
    output = Path(output).resolve()
    output.mkdir(exist_ok=False)
    owner = (output / 'owner.lock').open('a')
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(output / 'input.json', spec)

    def status(phase, **extra):
        temp = output / 'status.tmp'
        temp.write_text(json.dumps(dict(phase=phase, pid=os.getpid(), unix=time.time(), **extra), indent=2)+'\n')
        temp.replace(output / 'status.json')

    def command(module, args, phase):
        check_pins(spec['pins'])
        status(phase)
        invoke(module, args, output / (phase+'.log'))

    try:
        for i, dependency in enumerate(spec['dependencies']):
            status('waiting_original_dependency', index=i, dependency=dependency)
            wait_dependency(dependency)
            check_pins(spec['pins'])
        source = output / 'source-package'
        command('prepare_context_stage2_source', ['--spec', spec['source_spec'], '--output', source], 'bind-stage1')
        source_digest = sha(source / 'PACKAGE.json')
        common = read(spec['source_spec'])
        for entry in spec['runs']:
            algorithm = entry['algorithm']
            package = output / (algorithm+'-package')
            command('prepare_context_formal_package', [
                '--source-package', source, '--source-manifest-sha256', source_digest,
                '--source-specification', spec['source_spec'],
                '--acceptance-specification', spec['acceptance_spec'],
                '--algorithm', algorithm, '--output', package,
                '--remote-root', entry['remote_root']], algorithm+'-publish')
            experiment = dict(package=str(package), manifest_sha256=sha(package/'PREPARED.json'),
                initial_package=str(source), experiment=common['experiment'],
                stage1_root=common['training_root'], stage1_proof=common['training_proof'],
                template=spec['selection_template'])
            path = output / (algorithm+'-experiment.json')
            write(path, experiment)
            command('run_context_formal_experiment', ['--spec', path, '--output',
                output/(algorithm+'-experiment'), '--launch'], algorithm+'-run')
            result = read(output/(algorithm+'-experiment/status.json'))
            if result['phase'] != 'complete_training_and_selection_curve':
                raise ValueError('complete curve required before next algorithm')
        status('both_algorithm_curves_collected', scope='Simple-task selection curves; TEST and transfer remain separate')
    except BaseException as exc:
        status('stopped_preserve_handles', error=repr(exc), restart=False)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    specification = read(args.spec)
    check_pins(specification['pins'])
    if args.launch:
        run(specification, args.output)
    else:
        print(json.dumps(dict(status='validated_queue_spec_only_no_dispatch', spec=specification), indent=2))
