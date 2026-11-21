"""Reuse completed numerical acceptance by pinned report and transport content.

Only completed prefix/recovery/tail outputs are reusable. This never caches
launches, validates a new result, or treats an unbound PASS flag as evidence.
"""
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

from multimodal_jev.baseline import file_hash
from tools.reports.prepare_objective_training_candidate import read_bound

STATUSES = dict(prefix='finite_objective_training_outputs_audited',
                recovery='finite_objective_recovery_audited', tail='stage2_objective_tail_audited')


def check_tree(root, expected, workers=4):
    root = Path(root)
    if root.is_symlink():
        raise ValueError('receipt tree symlink')
    paths = list(root.rglob('*'))
    if any(p.is_symlink() for p in paths):
        raise ValueError('receipt dependency symlink')
    actual = {str(p.relative_to(root)) for p in paths if p.is_file()}
    if actual != set(expected):
        raise ValueError('receipt tree membership changed: ' + str(root))
    def check(item):
        name, digest = item
        if Path(name).is_absolute() or '..' in Path(name).parts or file_hash(root/name) != digest:
            raise ValueError('receipt dependency changed: ' + str(root/name))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(check, expected.items()))


class CompletedStageReceipts:
    def __init__(self, entries, *, workers=4):
        if set(entries) != set(STATUSES):
            raise ValueError('complete prefix/recovery/tail receipt set required')
        self.entries = copy.deepcopy(entries)
        self.workers = workers
        self.reports = {}
        self.bounds = {}
        self.events = []
        self.verify()

    def verify(self):
        started = time.monotonic()
        for kind, entry in self.entries.items():
            bound = json.loads(read_bound(entry['audit_config']))
            proof = json.loads(read_bound(entry['report']))
            if proof['status'] != STATUSES[kind]:
                raise ValueError('completed numerical acceptance required')
            package = Path(bound['training_package']['output'])
            manifest = json.loads((package/'PREPARED.json').read_text())
            identity = file_hash(package/'PREPARED.json')
            if identity != bound['training_package']['manifest_sha256'] or identity != proof['manifest_sha256']:
                raise ValueError('accepted package identity differs')
            check_tree(package, dict(manifest['files'], **{'PREPARED.json': identity}), self.workers)
            if file_hash(bound['transport']) != proof['transport_sha256']:
                raise ValueError('accepted transport changed')
            transport = json.loads(Path(bound['transport']).read_text())
            if transport['container_id'] != proof['container_id'] or transport['image'] != proof['image']:
                raise ValueError('accepted container identity differs')
            check_tree(bound['collection'], transport['files'], self.workers)
            if file_hash(bound['archive']) != transport['archive_sha256']:
                raise ValueError('accepted archive changed')
            for name in ('launch', 'inspect', 'wait'):
                read_bound(entry[name])
                if entry[name]['path'] != bound[name]:
                    raise ValueError('receipt handle path differs')
            inspection = json.loads(Path(bound['inspect']).read_text())
            if inspection['Id'] != proof['container_id'] or inspection['State']['ExitCode'] != 0 or inspection['State']['OOMKilled']:
                raise ValueError('completed non-OOM container required')
            if kind == 'recovery' and (not proof['fresh_process_npu_resume_verified'] or not all(proof['replay'][k] for k in (
                    'exact_masters','exact_adam','exact_batch','exact_bf16_execution','exact_sampled_groups',
                    'exact_reference_and_behavior','exact_objective_traces'))):
                raise ValueError('exact recovery proof required')
            self.bounds[kind], self.reports[kind] = bound, proof
        if self.bounds['recovery']['training_package']['parent_audit'] != self.bounds['prefix']:
            raise ValueError('recovery parent differs')
        if self.bounds['tail']['deployment']['acceptance'] != {k:self.bounds[k] for k in ('prefix','recovery')}:
            raise ValueError('tail lineage differs')
        self.events.append(dict(action='verify_completed_content', seconds=time.monotonic()-started,
                                numerical_replays=0, stages=list(STATUSES)))

    def audit(self, kind, bound, provenance):
        if bound != self.bounds[kind]:
            raise ValueError('unaccepted stage requested')
        return dict(copy.deepcopy(self.reports[kind]), provenance=provenance)

    def prefix_package(self, spec, provenance):
        if spec != self.bounds['prefix']['training_package']:
            raise ValueError('unaccepted prefix package requested')
        return copy.deepcopy(self.reports['prefix']['package_audit'])
