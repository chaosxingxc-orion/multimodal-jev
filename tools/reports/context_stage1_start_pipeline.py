"""Deterministic Stage1 prefix/replay/tail admission then formal training.

Config supplies versioned packages and immutable CPU/cache evidence. One
controller uses existing launch/collect/audit functions; failures terminate
without retrying or invoking a model. Formal launch waits until completion.
"""
import argparse,hashlib,json,subprocess,shlex,sys,time,fcntl
import math
from pathlib import Path
from multimodal_jev.baseline import file_hash, atomic_json
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config


def save_macro(path,key,value):
    commit_json(Path(path),dict(schema='jev-local-macros-v1',macros={key:value}))


def cpu_source(report, *, io_workers):
    from multimodal_jev.source_snapshot import verify
    from multimodal_jev.stage1_package_sources import runtime_inventory
    collection=Path(report['collection'])
    root=collection.parent/(collection.name.removesuffix('-collected')+'-package')
    files=verify(root,manifest_sha256=report['source_manifest_sha256'],io_workers=io_workers)
    return str(root),runtime_inventory(files)


def finish_formal(root, collected, audit_config, auditor):
    if (collected.get('status')!='collected_independent_training_audit_required'
            or collected.get('exit_code')!=0):
        raise ValueError('formal Stage1 collection failed: '+str(collected))
    proof=auditor(audit_config)
    if (proof.get('status')!='formal_infonce_artifacts_audited'
            or proof.get('container_id')!=collected['container_id'] or not proof.get('updates')):
        raise ValueError('matching independent formal Stage1 audit required')
    commit_json(root/'formal-audit.json',proof)
    commit_json(root/'terminal.json',dict(exit_code=0,status=proof['status'],result=collected,
        updates=len(proof['updates']),audit_sha256=file_hash(root/'formal-audit.json'),efficacy_evaluated=False))


def formal_audit_config(c,initial):
    template=Path(__file__).resolve().parents[2]/'configs/portable/operations/context-infonce-formal.json'
    return dict(operations_config=str(template),operations_local=c['formal_local'],initial_policy=initial)


def input_cache(c, root):
    if 'shared_cache_manifest_sha256' in c:
        from multimodal_jev.cached_context_preparation import processor_cache_identity
        package=c['package'];cache=Path(package['encoded_audio_cache'])
        manifest=cache.parent/'MANIFEST.json'
        if file_hash(manifest)!=c['shared_cache_manifest_sha256']:
            raise ValueError('shared input cache manifest changed')
        info=json.loads(manifest.read_text())
        design=json.loads(Path(package['design']).read_text())
        if (info['schema']!='context-full-learning-cache-v1'
                or info['recordings']!=design['stage1']['batch_config']['source_queries']
                or info['release_sha256']!=package['release_sha256']
                or info['prompt_layout']!=package['prompt_layout']
                or info['processor_sha256']!=processor_cache_identity(package['initial_policy_local'])
                or info['fresh_audio_checked']<=0 or info['max_tokens']>4096):
            raise ValueError('shared input cache must match complete TRAIN and policy processor')
        commit_json(root/'shared-cache-validated.json',dict(path=str(cache),sha256=file_hash(manifest)))
        return str(cache),str(manifest),file_hash(manifest)
    h=json.loads(Path(c['cache_launch']).read_text());job=h['config']
    code=subprocess.check_output(['ssh',job['host'],'docker wait '+h['container_id']],text=True).strip()
    observed=json.loads(subprocess.check_output(['ssh',job['host'],'docker inspect '+h['container_id']],text=True))[0]
    if code!='0' or observed['Image']!=job['image'] or observed['State']['ExitCode']!=0:raise ValueError('input cache failed')
    for name in ('MANIFEST.json','terminal.json'):
        data=subprocess.check_output(['ssh',job['host'],shlex.join(['cat',job['remote_root']+'/run/'+name])])
        (root/('cache-'+name)).write_bytes(data)
    cache=json.loads((root/'cache-MANIFEST.json').read_text())
    if cache['recordings']!=40513 or cache['fresh_audio_checked']!=32 or cache['max_tokens']>4096:raise ValueError('full cache acceptance failed')
    parent=job['remote_root']+'/run'
    return parent+'/cache',parent+'/MANIFEST.json',file_hash(root/'cache-MANIFEST.json')


def check_window(c, *, required=0):
    if 'not_after_unix' in c:
        deadline=c['not_after_unix']
        if type(deadline) not in (int,float) or not deadline>time.time()+required:
            raise ValueError('host allocation expires before this Stage1 window can finish')


def formal_lease(c):
    cutoff = c.get('not_after_unix')
    keys = ('lease_stop_grace_seconds', 'release_reserve_seconds', 'walltime_multiplier')
    if cutoff is None and 'not_after_unix' not in c and not any(key in c for key in keys):
        return None
    grace, reserve, multiplier = (c.get(key) for key in keys)
    if (type(cutoff) not in (int, float) or not math.isfinite(cutoff)
            or type(grace) is not int or grace < 1
            or type(reserve) is not int or reserve < grace + 30
            or type(multiplier) not in (int, float) or not math.isfinite(multiplier) or multiplier < 1):
        raise ValueError('explicit Stage1 deadline, stop grace, reserve and step multiplier required')
    return dict(lease_deadline_unix=cutoff, lease_stop_grace_seconds=grace)


def formal_lease_budget(c, lease, prepared, prefix, tail, *, now=None):
    steps = prepared['total_updates']
    timings = [update['slowest_rank_seconds'] for proof in (prefix, tail) for update in proof['updates']]
    if (len(prefix['updates']) != 2 or len(tail['updates']) != 2
            or type(steps) is not int or steps <= 0
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in timings)):
        raise ValueError('positive Stage1 prefix and tail step timings and full update count required')
    slowest = max(timings)
    multiplier = c['walltime_multiplier']
    seconds = math.ceil(slowest * steps * multiplier)
    now = time.time() if now is None else now
    reserve = c['release_reserve_seconds']
    return dict(**lease, slowest_step_seconds=slowest, planned_steps=steps, time_multiplier=multiplier,
                estimate_seconds=seconds, reserve_seconds=reserve, required_seconds=seconds+reserve, estimated_at_unix=now,
                fits_window=lease['lease_deadline_unix'] > now + seconds + reserve)


def run(c,provenance):
    cutoff = formal_lease(c)
    check_window(c)
    import torch
    torch.set_num_threads(1)
    from tools.reports import context_instruction_operations as finite
    from tools.reports import context_infonce_recovery as replay
    from tools.reports import context_infonce_tail_operations as tail
    from tools.reports import context_infonce_formal as formal
    from tools.reports.audit_context_infonce_training import audit_collected
    root=Path(c['output']);root.mkdir(parents=True,exist_ok=False)
    with (root/'owner.lock').open('w') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        stage='initialize'
        def progress(value):
            nonlocal stage
            stage=value
            atomic_json(root/'progress.json',dict(stage=stage,unix=time.time()))
        try:
            if 'design_sha256' in c:
                design=Path(c['package']['design']).read_bytes()
                if hashlib.sha256(design).hexdigest()!=c['design_sha256']:
                    raise ValueError('Stage1 experiment design changed')
                path=root/'design.json'
                with path.open('xb') as output:output.write(design)
                c=dict(c,package=dict(c['package'],design=str(path)))
            if c.get('cpu_template'):
                progress('waiting_for_fixed_image_cpu')
                cpu,_=load_config(c['cpu_template'],c['cpu_local'])
                handle=json.loads((Path(cpu['output'])/(cpu['tag']+'-launch.json')).read_text())
                code=subprocess.check_output(['ssh',cpu['host'],shlex.join(['docker','wait',handle['container_id']])],text=True).strip()
                if code!='0':raise ValueError('fixed-image CPU acceptance failed')
                if not Path(cpu['report']).exists():
                    subprocess.run([sys.executable,'-B','-m','tools.reports.collect_fixed_cpu_tests',
                        '--config',c['cpu_template'],'--local',c['cpu_local']],check=True)
            progress('verifying_fixed_image_source')
            cpu_bytes=Path(c['cpu_report']).read_bytes()
            cpu=json.loads(cpu_bytes)
            if (cpu['counts']!=dict(tests=c['cpu_expected_tests'],failures=0,errors=0,skipped=0)
                    or cpu['image']!=finite.lifecycle.IMAGE or cpu['state']['ExitCode']!=0
                    or cpu['state']['Status']!='exited' or cpu['terminal']['exit_code']!=0):
                raise ValueError('CPU acceptance failed')
            project,inventory=cpu_source(cpu,io_workers=c['io_workers'])
            # Exact event-driven wait, no model or frequent polling.
            progress('waiting_for_input_cache')
            cache,cache_manifest,cache_sha=input_cache(c,root)
            if 'wait_for_container' in c:
                h=json.loads(Path(c['wait_for_container']).read_text())
                if h['host']!=c['deployment']['host'] or h['image']!=finite.lifecycle.IMAGE:
                    raise ValueError('exact same-host resource predecessor required')
                progress('waiting_for_previous_device_owner')
                subprocess.check_output(['ssh',h['host'],shlex.join(['docker','wait',h['container_id']])],
                                        text=True)
                with Path(c['deployment']['lease']).open('r') as lease:
                    fcntl.flock(lease,fcntl.LOCK_EX)
            progress('finite_prepare')
            package=dict(c['package']);package['project']=project
            package['encoded_audio_cache']=cache
            package['external_pins']={k:v for k,v in package['external_pins'].items() if not k.endswith('/context-learning-cache-v1/run/MANIFEST.json')}
            package['external_pins'][cache_manifest]=cache_sha
            prep=finite.prepare(package,provenance,tested_runtime=inventory)
            common=dict(c['deployment'],cpu_report=c['cpu_report'],cpu_report_sha256=hashlib.sha256(cpu_bytes).hexdigest(),cpu_expected_tests=c['cpu_expected_tests'])
            ops=dict(common,**c['finite'],manifest_sha256=prep['manifest_sha256'])
            save_macro(c['finite_local'],'INSTRUCTION_OPERATIONS_CONFIG',ops)
            check_window(c);progress('finite_npu_launch');finite.launch(ops,provenance)
            audit_config=dict(operations_config=c['finite_template'],operations_local=c['finite_local'],initial_policy=package['initial_policy_local'])
            progress('finite_audit');prefix=audit_collected(audit_config);parent_audit=root/'finite-audit.json';commit_json(parent_audit,prefix)
            parent=dict(parent_operations_template=c['finite_template'],parent_operations_local=c['finite_local'],parent_audit=str(parent_audit),parent_audit_sha256=file_hash(parent_audit))
            rp=dict(common,**c['replay'],**parent)
            progress('replay_prepare');rp.update(replay.prepare(rp,provenance));rp.pop('status',None);rp.pop('formal_training_allowed',None)
            save_macro(c['replay_local'],'INFONCE_RECOVERY_CONFIG',rp)
            check_window(c);progress('replay_npu_launch');replay.launch(rp,provenance)
            progress('replay_audit');proof=replay.audit(dict(audit_config,operations_config=c['recovery_template'],operations_local=c['replay_local']))
            ra=root/'replay-audit.json';commit_json(ra,proof)
            tl=dict(common,**c['tail'],**parent,replay_audit=str(ra),replay_audit_sha256=file_hash(ra),tail_cpu_report=c['cpu_report'],tail_cpu_report_sha256=common['cpu_report_sha256'],tail_cpu_expected_tests=c['cpu_expected_tests'])
            progress('tail_prepare');tl.update(tail.prepare(tl,provenance,tested_source=(project,inventory)));tl.pop('status',None)
            save_macro(c['tail_local'],'INFONCE_RECOVERY_CONFIG',tl)
            check_window(c);progress('tail_npu_launch');tail.launch(tl,provenance)
            progress('tail_audit');tail_proof=tail.audit(dict(audit_config,operations_config=c['recovery_template'],operations_local=c['tail_local']))
            ta=root/'tail-audit.json';commit_json(ta,tail_proof)
            fo=dict(common,**c['formal'],**parent,replay_audit=str(ra),replay_audit_sha256=file_hash(ra),tail_audit=str(ta),tail_audit_sha256=file_hash(ta),design=package['design'],replay_operations_template=c['recovery_template'],replay_operations_local=c['replay_local'],tail_operations_template=c['recovery_template'],tail_operations_local=c['tail_local'],formal_training_dispatch_allowed=True)
            if fo.keys() & {'lease_deadline_unix', 'lease_stop_grace_seconds'}:
                raise ValueError('Stage1 lease must be configured at pipeline level')
            if cutoff is not None:
                fo.update(cutoff)
            progress('formal_prepare');prepared=formal.prepare(fo,provenance);fo['manifest_sha256']=prepared['manifest_sha256']
            save_macro(c['formal_local'],'INFONCE_FORMAL_CONFIG',fo)
            commit_json(root/'formal-admitted.json',dict(config=fo,prepared=prepared))
            if cutoff is not None:
                budget = formal_lease_budget(c, cutoff, prepared, prefix, tail_proof)
                commit_json(root/'formal-lease-budget.json',budget)
                check_window(c,required=budget['estimate_seconds']+budget['reserve_seconds'])
            progress('formal_launch');result=formal.launch(fo,provenance)
            progress('formal_audit');finish_formal(root,result,formal_audit_config(c,package['initial_policy_local']),formal.audit)
        except BaseException as exc:
            commit_json(root/'terminal.json',dict(exit_code=1,stage=stage,error=repr(exc),restart=False,preserve_handles=True,unix=time.time()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True);p.add_argument('--detach',action='store_true');a=p.parse_args();c,pr=load_config(a.config,a.local)
    if a.detach:
        d=Path(c['output']+'-driver');d.mkdir(parents=True,exist_ok=False)
        argv=[sys.executable,'-B','-m','tools.reports.context_stage1_start_pipeline','--config',str(Path(a.config).resolve()),'--local',str(Path(a.local).resolve())]
        with (d/'driver.log').open('x') as f:child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        commit_json(d/'started.json',dict(pid=child.pid,process_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19],argv=argv,unix=time.time()))
        print(child.pid)
    else:run(c,pr)
