"""Opt-in bounded-kernel/acoustic worker using unchanged dual owner semantics.

No activation, batch capacity or process-device guard is replaced. Installation
does not admit the policy for TRAIN/reference/GRPO use.
"""
import hashlib
from pathlib import Path

PARENT_SHA = 'c1b839340106a362368743130acc99057885c4d315c3ed610d1cbe695da13152'
WORKER = 'tools.reports.context_bounded_rollout_worker.BoundedAcousticWorker'
ACTOR_ORDER = 'tools.reports.context_actor_order_rollout_worker.ActorOrderAcousticWorker'


def owner_class(source_root, *, shared_scope=False, scope_module=None, actor_order=False):
    path = Path(__file__).with_name('context_dual_rollout_candidate.py')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PARENT_SHA:
        raise ValueError('shared dual owner changed; revalidate binding')
    before = 'tools.reports.context_dual_worker_candidate.DualContextWorkerCandidate'
    source = raw.decode()
    if source.count(before) != 1:
        raise ValueError('one explicit dual worker binding required')
    source = source.replace(before, ACTOR_ORDER if actor_order else WORKER)
    ns = {'__name__': __name__+'.derived', '__file__': str(path)}
    exec(compile(source, '<bounded-dual-owner>', 'exec'), ns)
    return ns['owner_class'](source_root, shared_scope=shared_scope, scope_module=scope_module)
