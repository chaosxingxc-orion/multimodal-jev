"""Prepare an isolated objective source candidate; never grants NPU admission."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from types import FunctionType

from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports import prepare_context_cached_runtime as closure
from tools.reports.context_plugin_trainer import generate


ADDITIONS = ('src/multimodal_jev/objective_plugin.py',
             'tools/reports/context_plugin_trainer.py', 'extensions/stage2_objective.py')
TESTS = ('tests/test_objective_plugin.py', 'tests/test_objective_plugin_native.py',
         'tests/test_objective_plugin_recovery.py')
OVERLAY_FILES = set(ADDITIONS+TESTS) | {'src/multimodal_jev/objective_trace.py',
    'tools/reports/audit_objective_traces.py','tests/test_objective_trace.py'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(root):
    root=Path(root)
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('symlink in source evidence')
    return {str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}


def prepare(c, provenance):
    parent=Path(c['parent']);snapshot=Path(c['tested_source']);output=Path(c['output'])
    if (output.exists() or any(output.resolve().is_relative_to(p.resolve())
            or p.resolve().is_relative_to(output.resolve()) for p in (parent,snapshot))):
        raise ValueError('fresh disjoint candidate required')
    if digest(parent/'PREPARED.json')!=c['parent_sha256']:
        raise ValueError('parent manifest changed')
    manifest=json.loads((parent/'PREPARED.json').read_text())
    files=inventory(parent);files.pop('PREPARED.json')
    if files!=manifest['files']:raise ValueError('parent payload changed')
    if digest(c['cpu_report'])!=c['cpu_report_sha256']:
        raise ValueError('CPU report changed')
    report=json.loads(Path(c['cpu_report']).read_text())
    if (report['counts']!=dict(tests=25,failures=0,errors=0,skipped=0)
            or report['image']!=manifest['image'] or report['state']['ExitCode']!=0
            or report['state']['OOMKilled'] or report['terminal']['exit_code']!=0
            or digest(snapshot/'FILES.json')!=report['source_manifest_sha256']):
        raise ValueError('successful bound CPU source snapshot required')
    tested=json.loads((snapshot/'FILES.json').read_text())
    observed=inventory(snapshot);observed.pop('FILES.json')
    if observed!=tested:raise ValueError('tested source changed')
    inherited=inventory(parent/'source')
    if any(name in inherited for name in ADDITIONS):
        raise ValueError('extension must not overwrite inherited source')
    overlay=c.get('overlay');overlay_files={}
    if overlay is not None:
        if set(overlay)!={'project','files'} or not isinstance(overlay['files'],dict):
            raise ValueError('explicit overlay project and file hashes required')
        overlay_files=overlay['files']
        if not overlay_files or set(overlay_files)-OVERLAY_FILES or set(overlay_files)&set(inherited):
            raise ValueError('overlay may only contain new objective files and tests')
        for name,h in overlay_files.items():
            p=Path(overlay['project'])/name
            if p.is_symlink() or digest(p)!=h:raise ValueError('objective overlay changed: '+name)
    check_paths({'objective_source_candidate':output})
    output.mkdir(parents=True,exist_ok=False)
    source=output/'source';shutil.copytree(parent/'source',source)
    for name in ADDITIONS+TESTS:
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(snapshot/name,target)
    for name in overlay_files:
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(Path(overlay['project'])/name,target)
    # Resolve imports from the exact tested snapshot, keeping inherited files.
    # Imports in generated source find the already copied extension; recursively
    # fill only missing dependencies, never overwrite the declared overlay.
    close=FunctionType(closure.close_sources.__code__,dict(closure.close_sources.__globals__,
        ROOTS=(),generate=generate))
    close(snapshot,source)
    actual=inventory(source)
    if any(actual.get(n)!=h for n,h in inherited.items()):
        raise ValueError('inherited runtime changed')
    for name in ADDITIONS+TESTS:
        if actual.get(name)!=overlay_files.get(name,tested[name]):
            raise ValueError('extension differs from declared bytes')
    extra={n:h for n,h in actual.items() if n not in inherited}
    for name,h in extra.items():
        if overlay_files.get(name,tested.get(name))!=h:
            raise ValueError('unverified new dependency: '+name)
    result=dict(schema='jev-objective-source-candidate-v1',
        status='source_candidate_prepared_not_npu_admitted',parent_sha256=c['parent_sha256'],
        cpu_report_sha256=c['cpu_report_sha256'],files=actual,additions=extra,
        inherited_files=len(inherited),inherited_source_unchanged=True,
        overlay_files=overlay_files,combined_cpu_admitted=False,
        inference_compatibility_verified=False,npu_admitted=False,
        training_dispatch_allowed=False,config_provenance=provenance)
    commit_json(output/'SOURCE-CANDIDATE.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--local',required=True)
    args=parser.parse_args();c,provenance=load_config(args.config,args.local)
    result=prepare(c,provenance)
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','additions')},indent=2))
