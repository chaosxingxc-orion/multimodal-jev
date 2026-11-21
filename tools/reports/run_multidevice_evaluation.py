"""Shared CPU audit engine, optional exact predecessor wait, one evaluation.

No retries or model-driven monitoring. All NPU admission still happens in the
shared lifecycle; source audits share only content-authenticated batch replay.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from types import FunctionType,SimpleNamespace

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.reports.prepare_objective_training_candidate import read_bound
from tools.reports.context_isolated_source_audit import IsolatedSourceAuditEngine
from tools.reports.finalize_context_infonce import wait_exact
from tools.reports import prepare_multidevice_selection as preparation
from tools.reports import multidevice_selection_operations as operations


def scoped(engine):
    parent=SimpleNamespace(audit=engine.preparation().audit,path=preparation.parent_api.path)
    expected=FunctionType(preparation.expected.__code__,dict(preparation.expected.__globals__,parent_api=parent))
    prepare=FunctionType(preparation.prepare.__code__,dict(preparation.prepare.__globals__,expected=expected))
    audit=FunctionType(preparation.audit.__code__,dict(preparation.audit.__globals__,expected=expected))
    api=SimpleNamespace(prepare=prepare,audit=audit)
    validate=FunctionType(operations.validate.__code__,dict(operations.validate.__globals__,preparation=api))
    launch=FunctionType(operations.launch.__code__,dict(operations.launch.__globals__,validate=validate))
    audit_run=FunctionType(operations.audit.__code__,dict(operations.audit.__globals__,preparation=api))
    run=FunctionType(operations.run.__code__,dict(operations.run.__globals__,launch=launch,audit=audit_run))
    # operations.run delegates to old.run and supplies globals from its module;
    # give it the scoped callbacks through its own globals above.
    return api,SimpleNamespace(validate=validate,run=run,audit=audit_run)


def validate(c):
    from multimodal_jev.evaluation_performance_gate import require_performance_admission
    require_performance_admission(c)
    if c['monitor_interval_seconds']!=900:raise ValueError('900-second ordinary monitoring required')
    for b in c['pins']:read_bound(b)
    proof=json.loads(read_bound(c['cpu_report']))
    if (proof['counts']!=dict(tests=41,failures=0,errors=0,skipped=0)
            or proof['state']['ExitCode']!=0 or proof['state']['OOMKilled'] or proof['terminal']['exit_code']!=0):
        raise ValueError('fixed image CPU test evidence required')
    snapshot=Path(proof['collection'])/'package/FILES.json'
    files=json.loads(snapshot.read_text())
    if file_hash(snapshot)!=proof['source_manifest_sha256']:raise ValueError('CPU source manifest differs')
    for name in ('context_multidevice_evaluation.py','prepare_multidevice_selection.py','multidevice_selection_operations.py'):
        path=Path(__file__).with_name(name)
        if files.get('tools/reports/'+name)!=file_hash(path):raise ValueError('tested evaluation source changed')
    root=Path(c['output']);package=Path(c['package']['output']);ops=Path(c['deployment']['operations_root'])
    for a,b in ((root,package),(root,ops),(package,ops)):
        if a.resolve().is_relative_to(b.resolve()) or b.resolve().is_relative_to(a.resolve()):
            raise ValueError('disjoint evaluation records required')
    if c.get('predecessor'):
        prior=c['predecessor'];record=json.loads(read_bound(prior['driver']))
        if 'tools.reports.run_objective_acceptance' not in record['argv']:
            raise ValueError('exact editable-objective predecessor required')
    return proof


def run(c,provenance):
    cpu=validate(c);root=Path(c['output']);check_paths({'evaluation_sequence':root});root.mkdir(parents=True,exist_ok=False)
    commit_json(root/'config.json',c);commit_json(root/'provenance.json',provenance)
    try:
        with IsolatedSourceAuditEngine(cpu_workers=c['cpu_workers'],io_workers=c['io_workers']) as engine:
            package,api=scoped(engine);spec=dict(c['package'])
            if not spec.get('manifest_sha256'):
                spec['manifest_sha256']=package.prepare(spec,provenance)['manifest_sha256']
            commit_json(root/'package-config.json',spec)
            proof=package.audit(spec,provenance);commit_json(root/'package-audit.json',proof)
            ops=dict(c['deployment'],selection_package=spec,package=spec['output'],manifest_sha256=spec['manifest_sha256'])
            commit_json(root/'deployment.json',ops)
            m,config,_=api.validate(ops)
            commit_json(root/'validation.json',dict(status='multidevice_evaluation_validated_not_executed',
                argv=operations.command(ops,m,config),npu_executed=False))
            if c.get('predecessor'):
                prior=c['predecessor'];record=json.loads(read_bound(prior['driver']))
                commit_json(root/'waiting.json',dict(process=record,terminal=prior['terminal'],mechanism='exact_pidfd'))
                wait_exact(record,prior['terminal'])
                terminal=json.loads(Path(prior['terminal']).read_text())
                if terminal.get('exit_code')!=0 or terminal.get('status')!=prior['required_status']:
                    raise ValueError('predecessor failed; no automatic evaluation dispatch')
            validate(c)
            result=api.run(ops,provenance)
            commit_json(root/'result.json',result)
        commit_json(root/'terminal.json',dict(exit_code=0,status='multidevice_evaluation_completed_and_audited'))
        return result
    except BaseException as e:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(e),restart=False,preserve_handles=True));raise


def detach(template,local):
    c,_=load_config(template,local);validate(c)
    root=Path(c['output']).with_name(Path(c['output']).name+'-driver');check_paths({'driver':root});root.mkdir(parents=True,exist_ok=False)
    argv=[sys.executable,'-B','-m','tools.reports.run_multidevice_evaluation','--config',str(Path(template).resolve()),
        '--local',str(Path(local).resolve()),'--action','run']
    with (root/'driver.log').open('x') as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=child.pid,argv=argv,started_unix=time.time(),process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19])
    commit_json(root/'started.json',record);return record


if __name__=='__main__':
    import torch
    torch.set_num_threads(1)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['run','detach','validate'],default='validate');a=p.parse_args()
    c,provenance=load_config(a.config,a.local)
    if a.action=='detach':result=detach(a.config,a.local)
    elif a.action=='run':result=run(c,provenance)
    else:validate(c);result=dict(status='sequence_validated_not_executed')
    print(json.dumps(result,indent=2))
