"""Wait for one CPU test container, then shared bounded Stage1 acceptance.

No formal resume, no retry, no per-step polling or model invocation.
"""
import argparse,json,subprocess,shlex,sys,time
from pathlib import Path
from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.baseline import file_hash,atomic_json

def run(c,provenance):
    import torch
    torch.set_num_threads(1)
    from tools.reports.context_instruction_operations import launch
    from tools.reports.audit_context_infonce_training import audit_collected
    root=Path(c['output']);root.mkdir(parents=True,exist_ok=False)
    try:
        cpu,_=load_config(c['cpu_template'],c['cpu_local'])
        h=json.loads((Path(cpu['output'])/(cpu['tag']+'-launch.json')).read_text())
        code=subprocess.check_output(['ssh',cpu['host'],shlex.join(['docker','wait',h['container_id']])],text=True).strip()
        if code!='0':raise ValueError('CPU acceptance failed')
        # Independent collector may already have run in the interactive task.
        report=Path(cpu['report'])
        if not report.exists():
            subprocess.run([sys.executable,'-B','-m','tools.reports.collect_fixed_cpu_tests','--config',c['cpu_template'],'--local',c['cpu_local']],check=True)
        r=json.loads(report.read_text())
        if r['counts']!=dict(tests=cpu['expected_tests'],failures=0,errors=0,skipped=0):raise ValueError('CPU acceptance incomplete')
        ops,_=load_config(c['operations_template'],c['operations_pending'])
        ops.update(cpu_report_sha256=file_hash(report))
        resolved=root/'operations.json';commit_json(resolved,dict(schema='jev-local-macros-v1',macros={'INSTRUCTION_OPERATIONS_CONFIG':ops}))
        atomic_json(root/'progress.json',dict(stage='finite_npu_launch'))
        result=launch(ops,provenance);commit_json(root/'launch-result.json',result)
        if result['exit_code']!=0:raise ValueError('NPU capacity acceptance failed; keep failure evidence')
        proof=audit_collected(dict(operations_config=c['operations_template'],operations_local=str(resolved),initial_policy=c['initial_policy']))
        commit_json(root/'audit.json',proof)
        commit_json(root/'terminal.json',dict(exit_code=0,status='capacity_two_updates_audited_no_formal_resume'))
    except BaseException as exc:
        commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc)));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True);p.add_argument('--detach',action='store_true');a=p.parse_args();c,pr=load_config(a.config,a.local)
    if a.detach:
        d=Path(c['output']+'-driver');d.mkdir(parents=True,exist_ok=False)
        argv=[sys.executable,'-B','-m','tools.reports.run_stage1_capacity_acceptance','--config',str(Path(a.config).resolve()),'--local',str(Path(a.local).resolve())]
        with (d/'driver.log').open('x') as f:child=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        commit_json(d/'started.json',dict(pid=child.pid,argv=argv,process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19],unix=time.time()));print(child.pid)
    else:run(c,pr)
