"""Run the already prepared six-request HF/vLLM diagnostic in fresh processes.

This is inference acceptance only. No optimizer, dataset scoring or dispatch.
The caller supplies an exclusively leased NPU and an immutable package.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def verify(package):
    package = Path(package)
    files = json.loads((package / 'FILES.json').read_text())
    actual = {}
    for path in package.rglob('*'):
        if path.is_symlink():
            raise ValueError('Package symlinks are forbidden')
        if path.is_file() and path != package / 'FILES.json':
            actual[str(path.relative_to(package))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != files:
        raise ValueError('Complete immutable package required')
    return hashlib.sha256((package / 'FILES.json').read_bytes()).hexdigest()


def run(package, output):
    from multimodal_jev.local_runtime_storage import require_training_storage
    from multimodal_jev.local_checkpoint import commit_json
    package, output = Path(package).resolve(), Path(output).resolve()
    if output.is_relative_to(package) or package.is_relative_to(output):
        raise ValueError('Output must be separate from package')
    digest = verify(package)
    require_training_storage(output)
    output.mkdir(exist_ok=False)
    started = time.time()
    phase = 'hf'
    try:
        for phase in ('hf', 'vllm'):
            verify(package)
            argv = [sys.executable, '-m', 'tools.reports.probe_context_attribute_parity',
                    '--config', str(package / 'probe-config.json'),
                    '--output', str(output), '--phase', phase]
            with (output / (phase + '.log')).open('x') as log:
                subprocess.run(argv, cwd=package, stdout=log, stderr=subprocess.STDOUT, check=True)
        from tools.reports.probe_context_attribute_parity import compare
        result = compare(json.loads((output / 'hf.json').read_text()),
                         json.loads((output / 'vllm.json').read_text()))
        if (result != json.loads((output / 'parity.json').read_text())
                or not result['exact'] or not result['finite_log_policy_supported']):
            raise ValueError('Exact finite six-request parity required')
        verify(package)
        commit_json(output / 'terminal.json', dict(exit_code=0, started_unix=started,
            finished_unix=time.time(), source_manifest_sha256=digest,
            scope='HF/vLLM inference only; no distributed update, restore or efficacy admission'))
        return result
    except BaseException as exc:
        commit_json(output / 'terminal.json', dict(exit_code=1, phase=phase,
            started_unix=started, finished_unix=time.time(),
            source_manifest_sha256=digest, error=repr(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.package, args.output)
