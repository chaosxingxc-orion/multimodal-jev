"""Content-bound reuse of the unchanged full reference journal checker.

Only the leaf proof is reused. Endpoint, release, independent input receipts,
and input-to-record binding are reconstructed by the original outer auditor
on every call. Runtime/transport/parity admission remains caller-owned.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
from types import FunctionType

from multimodal_jev.audit_singleflight import AuditSingleFlight
from multimodal_jev.baseline import file_hash
from multimodal_jev.context_reference_audit import audit as original_audit
from multimodal_jev.context_reference_dataset import digest
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.parallel_evidence import inventory, workers
from tools.reports.context_parallel_batch_audit import read_bound


class ReferenceAuditEngine:
    def __init__(self, *, io_workers=4, max_entries=2):
        self.io_workers = workers(io_workers)
        self.store = AuditSingleFlight(max_entries=max_entries, max_running=1)
        self.events = []
        self.lock = threading.Lock()

    def fingerprint(self, root, journal, arguments):
        project = Path(__file__).resolve().parents[2]
        paths = sorted({str(p) for folder in
                        (project/'src/multimodal_jev', project/'tools/reports', project/'tools/jev')
                        for p in folder.glob('*.py')})
        with ThreadPoolExecutor(max_workers=self.io_workers) as pool:
            code = dict(zip(paths, pool.map(file_hash, paths), strict=True))
        # Include directories too: an empty extra macro must invalidate reuse.
        files = {}
        for name, path in [('dataset', root), ('journal', journal)]:
            base = Path(path)
            files[name] = dict(path=str(base.absolute()),
                files=inventory(base, io_workers=self.io_workers),
                directories=sorted(str(p.relative_to(base)) for p in base.rglob('*') if p.is_dir()))
        return digest(dict(files=files, arguments=arguments, code=code))

    def audit(self, root, journal, **arguments):
        start = time.perf_counter()
        key = self.fingerprint(root, journal, arguments)
        verified = time.perf_counter()
        def compute():
            proof = original_audit(root, journal, **arguments)
            if self.fingerprint(root, journal, arguments) != key:
                raise ValueError('reference evidence changed during audit')
            return proof
        proof, hit = self.store.get(key, compute)
        # Check both cache hits and producers before handing a proof to callers.
        if self.fingerprint(root, journal, arguments) != key:
            raise ValueError('reference evidence changed before proof consumption')
        with self.lock:
            self.events.append(dict(content_key=key, cache_hit=hit,
                initial_content_verification_seconds=verified-start,
                elapsed_seconds=time.perf_counter()-start))
        return proof

    def replay(self, config, provenance):
        from tools.reports.audit_context_reference_dataset import run
        publications = []
        def capture(path, value):
            if str(path) != str(config['output']) or publications:
                raise ValueError('unexpected reference audit publication')
            publications.append(value)
        replay = FunctionType(run.__code__, dict(run.__globals__,
                              audit=self.audit, commit_json=capture))
        result = replay(config, provenance)
        if publications != [result]:
            raise ValueError('complete reference proof publication required')
        return result

    def results_auditor(self):
        """Explicit adapter; original runtime/transport/package checks stay intact."""
        from tools.reports.prepare_context_reference_job import audit_results
        return FunctionType(audit_results.__code__, dict(audit_results.__globals__,
                            replay_reference_outputs=self.replay))


def run(config, provenance):
    root = Path(config['output'])
    root.mkdir(parents=True, exist_ok=False)
    commit_json(root/'config.json', config)
    commit_json(root/'provenance.json', provenance)
    try:
        for path, expected in config['pins'].items():
            if file_hash(path) != expected:
                raise ValueError('reference audit code changed: '+path)
        parent = read_bound(config['reference_audit_config'])
        expected = read_bound(config['expected_output_audit'])
        from tools.reports.prepare_context_reference_job import collected_audit_config
        spec = dict(parent['reference_job'])
        spec.update({k: parent[k] for k in ('collection', 'results_audit_output')})
        leaf = collected_audit_config(spec)
        engine = ReferenceAuditEngine(io_workers=config['io_workers'])
        times = []
        for i in range(2):
            start = time.perf_counter()
            proof = engine.replay(leaf, expected['provenance'])
            times.append(time.perf_counter()-start)
            if proof != expected:
                raise ValueError('full reference output proof differs from original')
            commit_json(root/f'reference-proof-{i}.json', proof)
        if engine.store.executions != 1 or engine.store.hits != 1:
            raise ValueError('identical reference evidence must share one leaf replay')
        report = dict(status='shared_reference_output_replay_verified',
            full_output_proof_equal=True, recordings=proof['recordings'],
            macro_batches=proof['macro_batches'], output_replay_seconds=times,
            journal_computations=engine.store.executions, journal_reuses=engine.store.hits,
            events=engine.events, model_calls=0, npu_executed=False,
            runtime_transport_reaudited=False, training_dispatch_allowed=False)
        commit_json(root/'report.json', report)
        commit_json(root/'terminal.json', dict(exit_code=0, status=report['status']))
        return report
    except BaseException as exc:
        commit_json(root/'terminal.json', dict(exit_code=1, error=repr(exc), restart=False))
        raise


if __name__ == '__main__':
    import torch
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--local', required=True)
    args = parser.parse_args()
    config, provenance = load_config(args.config, args.local)
    print(json.dumps(run(config, provenance), indent=2))
