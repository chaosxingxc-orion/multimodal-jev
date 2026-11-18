"""Reuse the pinned finite runner with the actual-logits measurement contract."""
import argparse
import hashlib
from pathlib import Path

RUNNER_SHA='d92450a38bb3d3318513b39e691c355d129fe2701e8a6e096eb4ba5901534b6f'


def runner_namespace(source_root):
    path=Path(source_root)/'tools/reports/run_context_batch_probe_candidate.py'
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=RUNNER_SHA:
        raise ValueError('finite runner source changed')
    source=data.decode()
    replacements={
        'from tools.reports.context_batch_measurement_candidate import compare':
        'from tools.reports.context_dual_measurement_candidate import compare_dual as compare',
        "package/'tools/reports/context_batch_measurement_candidate.py'":
        "package/'tools/reports/context_dual_measurement_candidate.py'"}
    for before,after in replacements.items():
        if source.count(before)!=1:raise ValueError('finite runner binding changed')
        source=source.replace(before,after)
    namespace={'__name__':__name__+'.derived'}
    exec(compile(source,'<dual-finite-probe>', 'exec'),namespace)
    return namespace


def run(package,output):
    return runner_namespace(package)['run'](package,output)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--package',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.package,a.output)
