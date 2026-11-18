"""Reuse unchanged owner lifecycle with an explicit alternative worker class."""
import hashlib
from pathlib import Path

PARENT_SHA='9eaed6cecfb04065d80bfdf7c976a2cf1d25131e91b496cc775acdfe4a0f3cf6'


def owner_class(source_root,*,acoustic):
    path=Path(__file__).with_name('context_dual_rollout_candidate.py')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PARENT_SHA:raise ValueError('pinned dual owner changed')
    source=raw.decode();before='tools.reports.context_dual_worker_candidate.DualContextWorkerCandidate'
    if source.count(before)!=1:raise ValueError('single worker binding required')
    worker='AcousticBatchWorker' if acoustic else 'AcousticBaselineWorker'
    source=source.replace(before,'tools.reports.context_acoustic_worker_candidate.'+worker)
    ns={'__name__':__name__+'.derived'}
    exec(compile(source,'<acoustic-owner>','exec'),ns)
    return ns['owner_class'](source_root)
