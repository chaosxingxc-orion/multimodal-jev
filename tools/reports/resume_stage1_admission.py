"""Resume after audited prefix/replay; keep frozen packages and rerun no passed NPU work."""
import argparse,json,subprocess,shlex,sys,time
from pathlib import Path
from multimodal_jev.local_config import load_config
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.baseline import file_hash,atomic_json
from tools.reports.context_stage1_start_pipeline import save_macro, finish_formal, formal_audit_config


def run(c,p):
    import torch
    torch.set_num_threads(1)
    from tools.reports import context_infonce_tail_operations as tail,context_infonce_formal as formal
    from tools.reports.context_infonce_recovery import validate as validate_replay
    root=Path(c['output']);root.mkdir(parents=True,exist_ok=False)
    def progress(stage):atomic_json(root/'progress.json',dict(stage=stage,unix=time.time()))
    try:
        cpu,_=load_config(c['cpu_template'],c['cpu_local'])
        handle=json.loads((Path(cpu['output'])/(cpu['tag']+'-launch.json')).read_text())
        progress('waiting_for_exact_package_cpu')
        code=subprocess.check_output(['ssh',cpu['host'],shlex.join(['docker','wait',handle['container_id']])],text=True).strip()
        if code!='0':raise ValueError('exact source CPU failed')
        if not Path(cpu['report']).exists():subprocess.run([sys.executable,'-B','-m','tools.reports.collect_fixed_cpu_tests','--config',c['cpu_template'],'--local',c['cpu_local']],check=True)
        old,_=load_config(c['prior_template'],c['prior_local'])
        rp,_=load_config(old['recovery_template'],old['replay_local']);validate_replay(rp)
        ra=Path(old['output'])/'replay-audit.json';proof=json.loads(ra.read_text())
        if proof['status']!='finite_infonce_replay_audited' or proof['manifest_sha256']!=rp['manifest_sha256'] or not proof['fresh_process_npu_resume_verified']:raise ValueError('actual replay admission required')
        tl,_=load_config(old['recovery_template'],old['tail_local'])
        # New operations binding, unchanged package/config/source. CPU evidence is
        # supplied for exact already sealed bytes; original evidence stays intact.
        tl.update(tail_cpu_report=cpu['report'],tail_cpu_report_sha256=file_hash(cpu['report']),tail_cpu_expected_tests=cpu['expected_tests'])
        tail_local=root/'tail-operations.json';save_macro(tail_local,'INFONCE_RECOVERY_CONFIG',tl)
        progress('tail_npu_launch');tail.launch(tl,p)
        ac=dict(operations_config=old['recovery_template'],operations_local=str(tail_local),initial_policy=old['package']['initial_policy_local'])
        progress('tail_audit');proof=tail.audit(ac);ta=root/'tail-audit.json';commit_json(ta,proof)
        common={k:tl[k] for k in ('cpu_report','cpu_report_sha256','cpu_expected_tests','parent_operations_template','parent_operations_local','parent_audit','parent_audit_sha256')}
        fo=dict(old['deployment'],**old['formal'],**common,replay_audit=str(ra),replay_audit_sha256=file_hash(ra),tail_audit=str(ta),tail_audit_sha256=file_hash(ta),design=old['package']['design'],replay_operations_template=old['recovery_template'],replay_operations_local=old['replay_local'],tail_operations_template=old['recovery_template'],tail_operations_local=str(tail_local),formal_training_dispatch_allowed=True)
        progress('formal_prepare');prepared=formal.prepare(fo,p);fo['manifest_sha256']=prepared['manifest_sha256']
        save_macro(c['formal_local'],'INFONCE_FORMAL_CONFIG',fo)
        commit_json(root/'formal-admitted.json',dict(config=fo,prepared=prepared))
        progress('formal_launch');result=formal.launch(fo,p)
        progress('formal_audit');finish_formal(root,result,formal_audit_config(c,ac['initial_policy']),formal.audit)
    except BaseException as exc:commit_json(root/'terminal.json',dict(exit_code=1,error=repr(exc)));raise

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--local',required=True);a.add_argument('--detach',action='store_true');x=a.parse_args();c,p=load_config(x.config,x.local)
    if not x.detach:run(c,p)
    else:
        d=Path(c['output']+'-driver');d.mkdir(parents=True,exist_ok=False)
        argv=[sys.executable,'-B','-m','tools.reports.resume_stage1_admission','--config',str(Path(x.config).resolve()),'--local',str(Path(x.local).resolve())]
        with (d/'driver.log').open('x') as f:child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        commit_json(d/'started.json',dict(pid=child.pid,argv=argv,process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19],unix=time.time()));print(child.pid)
