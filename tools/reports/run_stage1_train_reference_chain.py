"""Prepare and run endpoint TRAIN parity, then build and independently audit reference."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import fcntl
from functools import partial
from hashlib import sha256
import json
import multiprocessing
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from multimodal_jev.baseline import atomic_json, file_hash
from multimodal_jev.context_execution import width
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports import context_reference_job_operations as ref_ops
from tools.reports import context_train_parity_operations as parity_ops
from tools.reports import prepare_context_reference_job as reference
from tools.reports import prepare_context_train_parity_job as parity
from tools.reports import audit_context_train_parity as parity_results
from tools.reports.finalize_context_infonce import wait_exact


def wait_for_device_predecessors(c, root):
    """Wait for ownership to end, independently of the predecessor's CPU audit."""
    host = c['deployment']['host']
    if host.startswith('-') or not re.fullmatch(r'[\w.@-]+', host):
        raise ValueError('explicit valid target host required')
    occupied = set(c['parity']['scoring_devices'])
    seen = set()
    receipts = []
    for item in c.get('device_predecessors', []):
        raw = Path(item['path']).read_bytes()
        if sha256(raw).hexdigest() != item['sha256']:
            raise ValueError('device predecessor handle changed')
        handle = json.loads(raw)
        cid, argv = handle['container_id'], handle['argv']
        if (handle['host'] != host or not re.fullmatch(r'[a-f0-9]{64}', cid) or cid in seen
                or not any(f'/dev/davinci{d}:/dev/davinci{d}' in argv for d in occupied)):
            raise ValueError('exact same-host device predecessor required')
        seen.add(cid)
        wait = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host,
                               shlex.join(['docker', 'wait', cid])],
                              text=True, capture_output=True, check=True)
        code = wait.stdout.strip()
        if not code.isdigit():
            raise ValueError('exact container exit status missing')
        receipts.append(dict(container_id=cid, exit_code=int(code), launch_sha256=item['sha256']))
    commit_json(root/'devices-released.json', receipts)


def prepare_reference(spec, provenance):
    import torch
    torch.set_num_threads(1)
    started = time.monotonic()
    result = reference.prepare(spec, provenance)
    return result['manifest_sha256'], time.monotonic() - started


def parity_result_auditor(session, parent_auditor):
    def audit(spec, provenance):
        # Validate the upstream proof separately; captures cannot be nested.
        parent = spec['parity_package']['parent']
        proof = parent_auditor(parent, provenance)

        def exact_parent(request, p):
            if request != parent:
                raise ValueError('unexpected parity parent')
            return dict(proof, provenance=p)

        key = ('train_parity', json.dumps(spec, sort_keys=True))
        checked = session.get(key, lambda: parity_results.audit(spec, provenance,
            package_auditor=partial(parity.audit, parent_auditor=exact_parent)))
        return dict(checked, provenance=provenance)

    return audit


def run(c, provenance, *, parent_auditor=None, proof_session=None):
    prep_workers = None
    if 'package_preparation' in c:
        prep_workers = width(c['package_preparation']['cpu_workers'])
        if prep_workers < 2:
            raise ValueError('parallel package preparation requires at least two CPU workers')
    root = Path(c['output'])
    check_paths({'reference controller': root})
    root.mkdir(exist_ok=False)
    with (Path(c['chain_lock'])).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def progress(stage):
            atomic_json(root/'progress.json', dict(stage=stage))

        try:
            parent = c['parent']
            progress('waiting_for_endpoint_runtime')
            manifest = Path(parent['output'])/'PREPARED.json'
            if c.get('parent_driver') is None:
                if file_hash(manifest) != parent['manifest_sha256']:
                    raise ValueError('explicit prepared endpoint runtime identity required')
            else:
                wait_exact(c['parent_driver'], manifest)
                parent = dict(parent, manifest_sha256=file_hash(manifest))
            rs = dict(c['reference'], parent=parent)
            ps = dict(c['parity'], parent=parent)
            with ExitStack() as preparation:
                future = None
                if prep_workers is not None:
                    progress('preparing_train_parity_and_reference')
                    pool = preparation.enter_context(ProcessPoolExecutor(
                        max_workers=prep_workers, mp_context=multiprocessing.get_context('spawn')))
                    future = pool.submit(prepare_reference, rs, provenance)
                else:
                    progress('preparing_train_parity')
                panel_digest = parity.prepare(ps, provenance, parent_auditor=parent_auditor)['manifest_sha256']
                ps.update(manifest_sha256=panel_digest)
                if c.get('device_predecessors'):
                    progress('waiting_for_device_predecessors')
                    wait_for_device_predecessors(c, root)
                progress('train_parity_launch_collect_audit')
                po = dict(c['deployment'], **c['parity_operations'], parity_package=ps,
                          package=ps['output'], manifest_sha256=ps['manifest_sha256'])
                parity_auditor = None
                if proof_session is None:
                    parity_ops.run_and_audit(po, provenance)
                else:
                    package_auditor = partial(parity.audit, parent_auditor=parent_auditor)
                    parity_auditor = parity_result_auditor(proof_session, parent_auditor)
                    parity_ops.run_and_audit(po, provenance, package_auditor=package_auditor,
                                             result_auditor=parity_auditor)
                progress('waiting_for_prepared_reference' if future is not None else 'preparing_reference')
                started = time.monotonic()
                digest, seconds = (future.result() if future is not None else prepare_reference(rs, provenance))
                rs.update(manifest_sha256=digest)
                commit_json(root/'reference-preparation.json', dict(overlapped_with_parity=future is not None,
                    prepare_seconds=seconds, wait_after_parity_seconds=time.monotonic()-started,
                    manifest_sha256=digest, npu_executed=False))
            parity_audit = json.loads((Path(po['operations_root'])/'finalization/audit-config.json').read_text())
            ro = dict(c['deployment'], **c['reference_operations'], reference_job=rs,
                      package=rs['output'], manifest_sha256=rs['manifest_sha256'],
                      parity_audit=parity_audit)
            progress('reference_launch_collect_audit')
            result = (ref_ops.run_and_audit(ro, provenance, parity_auditor=parity_auditor,
                      package_auditor=partial(reference.audit,parent_auditor=parent_auditor))
                      if parity_auditor else ref_ops.run_and_audit(ro, provenance))
            commit_json(root/'terminal.json', dict(exit_code=0, status=result['status'],
                        recordings=result['output_audit']['recordings'], training_dispatched=False))
        except BaseException as exc:
            commit_json(root/'terminal.json', dict(exit_code=1, error=repr(exc), restart=False))
            raise


if __name__ == '__main__':
    import torch
    torch.set_num_threads(1)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--local', required=True)
    p.add_argument('--detach', action='store_true')
    a = p.parse_args()
    c, provenance = load_config(a.config, a.local)
    if a.detach:
        root = Path(c['output']+'-driver')
        check_paths({'TRAIN reference driver': root})
        root.mkdir(parents=True, exist_ok=False)
        argv = [sys.executable, '-B', '-m', __spec__.name, '--config', str(Path(a.config).resolve()),
                '--local', str(Path(a.local).resolve())]
        with (root/'driver.log').open('x') as log:
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        ticks = Path(f'/proc/{child.pid}/stat').read_text().split(') ', 1)[1].split()[19]
        record = dict(pid=child.pid, argv=argv, process_start_ticks=ticks)
        commit_json(root/'started.json', record)
        print(json.dumps(record))
    else:
        run(c, provenance)
