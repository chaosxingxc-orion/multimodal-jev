"""Two-step GRPO DP2 training using the common isolated-service lifecycle."""
import argparse
import json
from pathlib import Path
import re

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from tools.reports.context_macro_probe_operations import command as probe_command, run as run_services
from tools.reports.run_context_batch_probe_candidate import verify
from tools.reports.prepare_context_batch_probe_candidate import IMAGE
from tools.reports.context_macro_trainer_candidate import ALGORITHM, experiment_window

TORCHRUN_ARGS=['--nnodes=1','--node-rank=0','--nproc-per-node=2',
               '--master-addr=127.0.0.1','--master-port=29500']


def reusable_mounts(c,t,root=None):
    """Immutable dataset/audio inputs plus the retained fixed-reference cache.

    A retained cache lives outside every run root so one fill serves all arms;
    an arm-local ``root/reference-cache`` keeps the historical mount set.
    """
    root=root or c['remote_root']
    mounts=[(c['shared_root'],True),(str(Path(t['encoded_audio_cache']).parent),True),
            (t['encoded_audio_cache'],False),(t['audio_mirror_root'],True)]
    cache=t.get('reference_cache')
    if cache is not None and cache!=root+'/reference-cache':mounts.append((cache,False))
    return mounts


def validate(c, *, formal=False, command_builder=None):
    package=Path(c['prepared'])/'package'
    r=json.loads((package.parent/'PREPARED.json').read_text())
    t=json.loads((package/'training.json').read_text());root=c['remote_root']
    if formal:
        from tools.reports.context_macro_formal_candidate import contract
        contract(t)
        efficiency=False
    else:efficiency=experiment_window(t)=='efficiency'
    replay=t['start_update']==1
    initial_policy=t['source_policy_id'];initial_path=root+'/package/policy';initial_adapter=root+'/package/adapter'
    if replay:
        binding=json.loads((package/'REPLAY.json').read_text())
        if file_hash(package/'REPLAY.json')!=r.get('replay_sha256'):raise ValueError('replay parent binding changed')
        initial_policy=binding['initial_behavior_policy_id'];initial_path=root+'/package/resume/policy-0001'
        initial_adapter=root+'/package/resume/adapter-0001'
        if t.get('resume_root')!=root+'/package/resume':raise ValueError('sealed first-step recovery required')
    elif t['start_update']!=0 or t.get('resume_root'):raise ValueError('initial prefix or one-step replay only')
    status=('macro_formal_prepared_only' if formal else 'macro_efficiency_prepared_only' if efficiency else
            'macro_training_replay_prepared_only' if replay else 'macro_training_prefix_prepared_only')
    if (r['status']!=status or r['image']!=IMAGE
            or verify(package)!=c['manifest_sha256'] or r['files_sha256']!=c['manifest_sha256']
            or r['remote_root']!=root or r['training_dispatch_allowed'] is not False
            or not Path(root).is_absolute() or '..' in Path(root).parts or root in ('/','/data','/mnt')
            or r['ipc_root']!=root+'/ipc'):
        raise ValueError('sealed finite DP2 package required')
    # Recovery state may be spaced out; policy export stays per update.
    interval=t['checkpoint_interval']
    if (formal and interval!=64) or (not formal and (type(interval) is not int or not 1<=interval<=t['stop_update'])):
        raise ValueError('declared bounded checkpoint cadence required')
    # The fixed reference is either refreshed inside the run root (historical)
    # or precomputed once into a retained, disjoint cache bound to an audited
    # content index that every step must reproduce exactly.
    cache=t['reference_cache'];retained=cache!=root+'/reference-cache'
    if retained:
        path=Path(cache);index=package/'REFERENCE-INDEX.json'
        if (not path.is_absolute() or '..' in path.parts or ':' in cache
                or path.is_relative_to(root) or Path(root).is_relative_to(path)
                or index.is_symlink() or not index.is_file()
                or t.get('reference_index')!=str(index)
                or t.get('reference_index_sha256')!=file_hash(index)):
            raise ValueError('retained reference cache must be disjoint and indexed')
    if (t.get('reference_condition')=='precomputed_retained')!=retained:
        raise ValueError('reference condition must match the declared reference cache')
    if (t['physical_devices']!=[4,5] or t['global_batch']!=128 or t['microbatch'] not in ((8,16,32) if efficiency or formal else (8,))
            or t['algorithm']!=ALGORITHM or t['dual_arm']!='joint'
            or t['engineering_prefix'] is not (not formal)
            or t['training_dispatch_allowed'] is not formal or t['source_root']!=root+'/package'
            or t['model']!=c['model']
            or t['source_policy']!=root+'/package/policy' or t['initial_adapter']!=initial_adapter
            or t['reference_adapter']!=root+'/package/adapter' or t['dual_runtime_identity']!=r['runtime_identity']):
        raise ValueError('two-update DP2/global128/micro8 joint GRPO contract required')
    for key in ('tag','host'):
        if not re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_.-]*',c[key]):raise ValueError('safe host/tag required')
    client=t['macro_client']
    if (client['max_queries']!=128 or client['rollout_queries']!=32
            or client['runtime_identity']!=r['runtime_identity'] or client['journal_root']!=root+'/journal'):
        raise ValueError('complete macro contract required')
    for i in range(2):
        service=json.loads((package/f'service-{i}.json').read_text());endpoint=client['replicas'][i]
        if (service['policy']['physical_devices']!=[6+i]
                or service['policy']['snapshot']!=initial_path
                or service['policy']['adapter']!=initial_adapter
                or service['policy']['policy_id']!=initial_policy
                or service['policy']['contract']!=t['source_contract']
                or service['policy'].get('process_device_scope') is not None
                or service['dual_runtime_identity']!=t['dual_runtime_identity']
                or service['socket_path']!=endpoint['socket'] or service['exchange']!=endpoint['exchange']
                or not all(Path(p).is_relative_to(root+'/ipc') for p in endpoint.values())):
            raise ValueError('isolated matching dual services required')
    pins=json.loads((package/'FILES.json').read_text())
    for p,h in t['pins'].items():
        if Path(p).is_relative_to(root):
            if pins.get(str(Path(p).relative_to(root+'/package')))!=h:
                raise ValueError('internal dependency differs')
    if formal and command_builder is None:raise ValueError('explicit formal command builder required')
    for role in ('preflight','client','service-0','service-1'):(command_builder or command)(c,role)
    return r


def command(c,role):
    argv=probe_command(c,role);root=c['remote_root'];package=root+'/package'
    t=json.loads((Path(c['prepared'])/'package/training.json').read_text())
    # Services share only the sealed policy and local run directory. The actor
    # and CPU preflight additionally see the shared release and local audio cache.
    if role in ('client','preflight'):
        mounts=reusable_mounts(c,t)
        for p,_ in mounts:
            if (not Path(p).is_absolute() or ':' in p or '..' in Path(p).parts
                    or Path(p).is_relative_to(root) or Path(root).is_relative_to(p)):
                raise ValueError('reusable inputs and runtime must be disjoint')
        before=argv.index('--entrypoint')
        extra=[]
        for p,ro in mounts:extra+=['-v',p+':'+p+(':ro' if ro else '')]
        argv[before:before]=extra
        mapped=[root,c['model'],c['driver_root'],*[p for p,_ in mounts]]
        if any(not any(Path(p).is_relative_to(m) for m in mapped) for p in t['pins']):
            raise ValueError('dependency outside explicit mounts')
        if role=='client':
            for before,after in (('--cpus=8','--cpus=16'),('--memory=32g','--memory=96g'),
                                 ('--shm-size=2g','--shm-size=8g')):
                argv[argv.index(before)]=after
            before=argv.index('--entrypoint');devices=[]
            for name in ('davinci4','davinci5','davinci_manager','devmm_svm','hisi_hdc'):
                devices+=['--device','/dev/'+name+':/dev/'+name]
            argv[before:before]=devices
            argv=argv[:argv.index(IMAGE)+1]+['torchrun',*TORCHRUN_ARGS,
                package+'/tools/reports/context_macro_trainer_candidate.py','--config',package+'/training.json',
                '--output',root+'/run']
        else:
            code="""import json
from pathlib import Path
from tools.reports.context_macro_trainer_candidate import generate
from tools.reports.context_dual_actor_candidate import load
from tools.reports.context_dual_session_candidate import DualContextRolloutSession
from tools.reports.context_dual_runtime_candidate import verify_runtime
from tools.reports.run_context_batch_probe_candidate import verify
from multimodal_jev.baseline import file_hash
from multimodal_jev.span_runtime_profile import initialize_verl_preserving_qwen3
from multimodal_jev.context_policy_artifacts import read_execution_bundle,verify_native_adapter
root=Path(ROOT);c=json.loads((root/'training.json').read_text())
verify(root);verify_runtime(c);generate(root)
for p,h in c['pins'].items():
 if file_hash(p)!=h:raise ValueError('dependency changed: '+p)
from multimodal_jev.slurp_portable_release import load as load_release
from multimodal_jev.context_training_schedule import load_schedule
rows,task,_=load_release(c['release'],c['release_sha256'],'learning',purpose='optimize')
batches,_=load_schedule(c['schedule'],c['schedule_sha256'],release_sha256=c.get('schedule_release_sha256',c['release_sha256']),population={r['id']:r for r in rows})
assert len(rows)==40513 and len(batches)==634
initialize_verl_preserving_qwen3();load(root)
from verl.workers.actor.dp_actor import DataParallelPPOActor
b=read_execution_bundle(c['source_policy'],expected_policy_id=c['source_policy_id'],expected_contract=c['source_contract'],expected_base_id=c['base_id'])
verify_native_adapter(c['initial_adapter'],b)
import subprocess,sys
probe=Path('/jev-tmp/dp2-rendezvous-probe.py')
probe.write_text('import torch, torch.distributed as d\\nfrom datetime import timedelta\\nd.init_process_group("gloo",timeout=timedelta(seconds=30))\\nx=torch.tensor(float(d.get_rank()+1))\\nd.all_reduce(x)\\nassert d.get_world_size()==2 and x.item()==3.0\\nd.destroy_process_group()\\n')
subprocess.run([sys.executable,'-m','torch.distributed.run',*TORCHRUN_ARGS_LITERAL,str(probe)],check=True,timeout=90)
print('fixed-image DP2 actor/session/runtime/policy imports, pins and static-loopback Gloo rendezvous passed')
""".replace('ROOT',repr(package)).replace('TORCHRUN_ARGS_LITERAL',repr(TORCHRUN_ARGS))
            if t.get('start_update')==1:
                code=code.replace("verify_native_adapter(c['initial_adapter'],b)",
                    "verify_native_adapter(c['reference_adapter'],b)\n"
                    "resume=Path(c['resume_root'])/'policy-0001'\n"
                    "m=json.loads((resume/'MANIFEST.json').read_text())\n"
                    "rb=read_execution_bundle(resume,expected_policy_id=m['policy_id'],expected_contract=c['source_contract'],expected_base_id=c['base_id'])\n"
                    "verify_native_adapter(c['initial_adapter'],rb)")
            argv=argv[:argv.index(IMAGE)+1]+['python','-B','-c',code]
    return argv


def run(c,provenance):
    if c.get('finite_training_dispatch_allowed') is not True:raise ValueError('finite training disabled')
    return run_services(dict(c,finite_inference_dispatch_allowed=True),provenance,
        validator=validate,command_builder=command,required_devices=(4,5,6,7))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['validate','run'],default='validate');a=p.parse_args()
    c,provenance=load_config(a.config,a.local)
    print(json.dumps(run(c,provenance) if a.action=='run' else validate(c),indent=2))
