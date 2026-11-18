"""Measured, content-bound reuse of independent reference input reconstruction.

The original loader still authenticates receipt/proof bytes on every call.
Only its pure bind(proof, stream, rows, task, identities) result is shared.
No global module mutation, model call, or NPU dispatch.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import inspect
import json
from pathlib import Path
import threading
import time
from types import FunctionType

from multimodal_jev.audit_singleflight import AuditSingleFlight
from multimodal_jev.baseline import file_hash
from multimodal_jev.context_reference_dataset import digest
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from tools.reports import context_reference_input_evidence as inputs
from tools.reports.context_parallel_batch_audit import read_bound
from tools.reports.context_shared_reference_audit import ReferenceAuditEngine


class InputReuseEngine(ReferenceAuditEngine):
    def __init__(self, *, io_workers=4, max_entries=2):
        super().__init__(io_workers=io_workers, max_entries=max_entries)
        self.input_store = AuditSingleFlight(max_entries=max_entries, max_running=1)
        self.input_events = []
        self.phases = []
        self.measurement_lock = threading.Lock()

    def code_identity(self):
        project = Path(__file__).resolve().parents[2]
        paths = sorted(str(p) for folder in
                       (project/'src/multimodal_jev', project/'tools/reports', project/'tools/jev')
                       for p in folder.glob('*.py'))
        with ThreadPoolExecutor(max_workers=self.io_workers) as pool:
            return dict(zip(paths, pool.map(file_hash, paths), strict=True))

    def bind(self, proof, stream, **arguments):
        start = time.perf_counter()
        def fingerprint():
            return digest(dict(proof=proof, stream=stream, arguments=arguments,
                               code=self.code_identity()))
        key = fingerprint()
        verified = time.perf_counter()
        def compute():
            value = inputs.bind(proof, stream, **arguments)
            if fingerprint() != key:
                raise ValueError('independent input evidence changed during reconstruction')
            return value
        value, hit = self.input_store.get(key, compute)
        if fingerprint() != key:
            raise ValueError('independent input evidence changed before consumption')
        with self.measurement_lock:
            self.input_events.append(dict(cache_hit=hit, content_key=key,
                verification_seconds=verified-start, elapsed_seconds=time.perf_counter()-start))
        return value

    def load_inputs(self, config, **arguments):
        loader = FunctionType(inputs.load.__code__, dict(inputs.load.__globals__, bind=self.bind))
        return loader(config, **arguments)

    def timed(self, name, fn):
        def wrapped(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                with self.measurement_lock:
                    self.phases.append(dict(phase=name, seconds=time.perf_counter()-start))
        return wrapped

    def replay(self, config, provenance):
        from tools.reports.audit_context_reference_dataset import run
        from multimodal_jev.cached_context_preparation import processor_cache_identity
        publications = []
        def capture(path, value):
            if str(path) != str(config['output']) or publications:
                raise ValueError('unexpected reference publication')
            publications.append(value)
        source = inspect.getsource(run)
        replacements = {
            '    from tools.reports.context_reference_input_evidence import load as load_inputs,bindings_from_records':
            '    load_inputs,bindings_from_records = measured_inputs,measured_bindings',
            '    from multimodal_jev.cached_context_preparation import processor_cache_identity':
            '    processor_cache_identity = measured_processor',
        }
        for before, after in replacements.items():
            if source.count(before) != 1:
                raise ValueError('reference replay adapter anchor changed')
            source = source.replace(before, after)
        scope = dict(run.__globals__, commit_json=capture,
            load=self.timed('release_population', run.__globals__['load']),
            read_execution_bundle=self.timed('endpoint_bundle', run.__globals__['read_execution_bundle']),
            audit=self.timed('journal', self.audit),
            measured_inputs=self.timed('independent_inputs', self.load_inputs),
            measured_bindings=self.timed('record_bindings', inputs.bindings_from_records),
            measured_processor=self.timed('processor', processor_cache_identity))
        exec(compile(source, '<content-bound-reference-replay>', 'exec'), scope)
        result = scope['run'](config, provenance)
        if publications != [result]:
            raise ValueError('complete reference output proof required')
        return result


def run(config, provenance):
    root = Path(config['output'])
    root.mkdir(parents=True, exist_ok=False)
    commit_json(root/'config.json', config)
    commit_json(root/'provenance.json', provenance)
    try:
        for path, expected in config['pins'].items():
            if file_hash(path) != expected:
                raise ValueError('input reuse code changed: '+path)
        parent = read_bound(config['reference_audit_config'])
        expected = read_bound(config['expected_output_audit'])
        from tools.reports.prepare_context_reference_job import collected_audit_config
        spec = dict(parent['reference_job'])
        spec.update({k: parent[k] for k in ('collection', 'results_audit_output')})
        leaf = collected_audit_config(spec)
        engine = InputReuseEngine(io_workers=config['io_workers'])
        timings = []
        for index in range(2):
            start = time.perf_counter()
            phase_start = len(engine.phases)
            proof = engine.replay(leaf, expected['provenance'])
            timing = dict(seconds=time.perf_counter()-start, phases=engine.phases[phase_start:])
            timings.append(timing)
            if proof != expected:
                raise ValueError('full reference output proof differs from original')
            commit_json(root/f'reference-proof-{index}.json', proof)
            commit_json(root/f'timing-{index}.json', timing)
        if (engine.input_store.executions, engine.input_store.hits,
                engine.store.executions, engine.store.hits) != (1, 1, 1, 1):
            raise ValueError('one computation and reuse for both input and journal required')
        report = dict(status='shared_reference_inputs_and_outputs_verified',
            full_output_proof_equal=True, recordings=proof['recordings'], macro_batches=proof['macro_batches'],
            timings=timings, input_events=engine.input_events, journal_events=engine.events,
            input_computations=engine.input_store.executions, input_reuses=engine.input_store.hits,
            journal_computations=engine.store.executions, journal_reuses=engine.store.hits,
            model_calls=0, npu_executed=False, runtime_transport_reaudited=False,
            training_dispatch_allowed=False)
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
