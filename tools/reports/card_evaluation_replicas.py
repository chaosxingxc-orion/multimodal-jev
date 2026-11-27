"""Run independent exact evaluation workers under one physical-device lease."""
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import signal
import time

from multimodal_jev import evaluation_blocks as blocks
from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from tools.reports.tpo_evaluation_blocks import worker


_CACHE_VARIABLES = (
    'TMPDIR', 'HF_HOME', 'XDG_CACHE_HOME', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR',
    'ASCEND_WORK_PATH', 'ASCEND_CACHE_PATH', 'VLLM_RPC_BASE_PATH',
)


def replica_count(value):
    if type(value) is not int or value < 1:
        raise ValueError('replicas_per_device must be a positive integer')
    return value


def _replica(run_worker, spec, temporary):
    for key in _CACHE_VARIABLES:
        os.environ[key] = str(temporary)
    run_worker(spec)


def _run_processes(run_worker, jobs):
    temporary = Path(os.environ['TMPDIR'])
    context = mp.get_context('spawn')
    processes = []

    def interrupt(signum, frame):
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, interrupt)
    previous_interrupt = signal.signal(signal.SIGINT, interrupt)
    try:
        for name, spec in jobs:
            cache = temporary / name
            cache.mkdir()
            process = context.Process(target=_replica, args=(run_worker, spec, cache), name=name)
            process.start()
            processes.append(process)
        pending = {process.sentinel: process for process in processes}
        while pending:
            for sentinel in wait(pending):
                process = pending.pop(sentinel)
                process.join()
                if process.exitcode != 0:
                    raise RuntimeError(f'{process.name} exited with code {process.exitcode}')
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
            process.close()
        signal.signal(signal.SIGTERM, previous)
        signal.signal(signal.SIGINT, previous_interrupt)


def _run_workers(spec):
    count = replica_count(spec.get('replicas_per_device', 1))
    if count == 1:
        return worker(spec)
    if (len(spec['devices']) != 1 or 'single_card' not in spec or 'choice' in spec
            or any(k in spec for k in ('single_card_probe', 'single_card_reference',
                                        'finite_policy_parity'))):
        raise ValueError('replica concurrency requires one device and a shared single-card scoring queue')
    output = Path(spec['output'])
    output.mkdir()
    started = time.monotonic()
    jobs, receipts = [], []
    for index in range(count):
        work = output / f'replica-{index}'
        receipt = work.with_suffix('.json')
        jobs.append((work.name, dict(spec, output=str(work), result_path=str(receipt))))
        receipts.append(receipt)
    _run_processes(worker, jobs)
    rows = [blocks.read(receipt) for receipt in receipts]
    plan, = {row['plan_sha256'] for row in rows}
    completed = sorted((row for replica in rows for row in replica['blocks']), key=lambda r: r['block'])
    if (len({row['block'] for row in completed}) != len(completed)
            or any(row['devices'] != spec['devices'] for row in rows)):
        raise ValueError('replicas reported overlapping work or changed devices')
    result = dict(rows[0], plan_sha256=plan, blocks=completed, replicas_per_device=count,
                  seconds=time.monotonic() - started,
                  model_load_seconds=max((r['model_load_seconds'] or 0) for r in rows),
                  replica_blocks=[len(row['blocks']) for row in rows],
                  singleton_differences=sum(r.get('singleton_differences', 0) for r in rows))
    commit_json(Path(spec['result_path']), result)
    return result


def run(spec):
    replay = spec.get('replica_replay')
    if not replay:
        return _run_workers(spec)
    frozen = Path(replay['plan'])
    if file_hash(frozen) != replay['sha256']:
        raise ValueError('registered replay plan changed')
    manifest = blocks.read(frozen)
    root = Path(spec['queue']['root'])
    archive = Path(replay['archive'])
    original = blocks.checked_plan(archive, manifest['plan']['namespace'])
    expected = {b['key']: b for b in original['plan']['blocks']}
    chosen = manifest['plan']['blocks']
    if (not chosen or len({b['key'] for b in chosen}) != len(chosen)
            or any(expected.get(b['key']) != b for b in chosen)
            or {k: v for k, v in original['plan'].items() if k != 'blocks'}
            != {k: v for k, v in manifest['plan'].items() if k != 'blocks'}):
        raise ValueError('replay must use unchanged blocks from the registered evaluation')
    reference = {b['key']: blocks.read_block(archive, original, b) for b in chosen}
    root.mkdir()
    commit_json(root / 'plan.json', manifest)
    result = _run_workers(dict(spec, result_path=str(root / 'timing.json')))
    if {row['block'] for row in result['blocks']} != set(reference):
        raise ValueError('replay did not finish all registered blocks')
    differences = 0
    for block in chosen:
        actual = blocks.read_block(root, manifest, block)['prob']['records']
        archived = reference[block['key']]['prob']['records']
        differences += sum(a != b for x, y in zip(actual, archived, strict=True)
                           for a, b in zip(x['prob'], y['prob'], strict=True))
    result.update(probability_differences=differences, pairs=sum(sum(b['widths']) for b in chosen),
                  replicas_per_device=spec.get('replicas_per_device', 1),
                  reference_plan_sha256=original['sha256'])
    commit_json(Path(spec['result_path']), result)
    return result
