"""Prepare only a bounded integrated trainer/service package using local macros."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from multimodal_jev.local_config import load_config
from multimodal_jev.cached_context_preparation import processor_cache_identity
from tools.reports.context_dual_trainer_candidate import generate
from tools.reports.context_dual_rollout_candidate import RUNTIME
from multimodal_jev.context_logits_wire import WIRE
from tools.reports.prepare_context_batch_probe_candidate import TOOLS


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rebase(value,old,new):
    if isinstance(value,str):return value.replace(old,new)
    if isinstance(value,list):return [rebase(v,old,new) for v in value]
    if isinstance(value,dict):return {rebase(k,old,new):rebase(v,old,new) for k,v in value.items()}
    return value


def prepare(c,provenance):
    parent=Path(c['parent']);project=Path(c['project']);output=Path(c['output'])
    receipt=json.loads((parent/'PREPARED.json').read_text())
    if sha(parent/'PREPARED.json')!=c['parent_sha256']:raise ValueError('parent identity changed')
    actual={str(p.relative_to(parent)):sha(p) for p in parent.rglob('*') if p.is_file() and p!=parent/'PREPARED.json'}
    if actual!=receipt['files']:raise ValueError('parent payload changed')
    remote=c['remote_root'];old=receipt['remote_root']
    if (not Path(remote).is_absolute() or '..' in Path(remote).parts or remote==old
            or output.exists() or output.resolve().is_relative_to(parent.resolve())):
        raise ValueError('new isolated candidate paths required')
    original=json.loads((parent/'config.json').read_text())
    if original['source_policy_id']!=c['source_policy_id']:raise ValueError('fixed Stage1 source differs')
    policy=Path(c['local_policy']);manifest=json.loads((policy/'MANIFEST.json').read_text())
    if manifest['policy_id']!=c['source_policy_id']:raise ValueError('local source policy differs')
    processor_id=processor_cache_identity(policy)
    generate(project) # validate original trainer transformation before copying
    output.mkdir();shutil.copytree(parent/'source',output/'source')
    additions=['context_dual_trainer_candidate.py','context_dual_runtime_candidate.py','context_dual_actor_candidate.py',
        'context_dual_session_candidate.py','context_probability_policy_v2.py','context_dual_rollout_candidate.py',
        'context_dual_worker_candidate.py','serve_context_dual_candidate.py','serve_context_batched_candidate.py']
    for name in set(TOOLS+additions):
        dest=output/'source/tools/reports'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(project/'tools/reports'/name,dest)
    for name in ('context_dual_credit.py','context_logits_wire.py','context_native_actor.py','context_lora_checkpoint_scope.py'):
        shutil.copy2(project/'src/multimodal_jev'/name,output/'source/src/multimodal_jev'/name)
    # Parent trainer contains the old arithmetic adaptation. Derive from the
    # pristine pinned trainer and integrate through the explicit new entrypoint.
    for name in ('train_context_online_candidate.py','context_switch_rollout.py'):
        shutil.copy2(project/'tools/jev'/name,output/'source/tools/jev'/name)
    generate(output/'source')
    from tools.reports.context_batched_owner_candidate import transformed_source
    transformed_source(output/'source') # reject incompatible inherited owner before sealing
    files={str(p.relative_to(output/'source')):sha(p) for p in (output/'source').rglob('*.py') if p.is_file()}
    (output/'RUNTIME-SOURCES.json').write_text(json.dumps(files,sort_keys=True,indent=2)+'\n')
    identity=dict(image_sha256=receipt['image'].removeprefix('sha256:'),source_manifest_sha256=sha(output/'RUNTIME-SOURCES.json'),
        processor_sha256=processor_id,profile=RUNTIME,output_protocol=WIRE)
    config=rebase(original,old,remote);config.pop('binary_policy_arithmetic',None)
    config.update(source_root=remote+'/source',dual_runtime_manifest=remote+'/RUNTIME-SOURCES.json',
        dual_runtime_identity=identity,algorithm='dual_credit_rloo_clipped_v1',dual_arm='joint',
        reference_cache=c['shared_reference_root'],start_update=0,stop_update=2,engineering_prefix=True,training_dispatch_allowed=False)
    service=rebase(json.loads((parent/'service-config.json').read_text()),old,remote)
    service.update(source_root=config['source_root'],source_policy=config['source_policy'],
        dual_runtime_manifest=config['dual_runtime_manifest'],dual_runtime_identity=identity,transport_protocol='bulk_dual_logits_candidate_v1')
    service['policy'].update(candidate_runtime=RUNTIME,output_protocol=WIRE,max_batch_queries=4,max_batch_tokens=16384)
    external={p:h for p,h in original['pins'].items() if not p.startswith(old+'/')}
    localpins={remote+'/'+str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()}
    config['pins']=dict(external,**localpins);service['pins']=dict(config['pins'])
    for name,value in [('config.json',config),('service-config.json',service)]:
        (output/name).write_text(json.dumps(value,indent=2)+'\n')
    runner=(parent/'run.py').read_text()
    for before,after in [('source/tools/jev/serve_context_rollout.py','source/tools/reports/context_dual_runtime_candidate.py'),
                         ('source/tools/jev/train_context_online_candidate.py','source/tools/reports/context_dual_trainer_candidate.py')]:
        if runner.count(before)!=1:raise ValueError('supervisor command binding changed')
        runner=runner.replace(before,after)
    # tools.reports imports require the common source root in both processes.
    before=" env=dict(os.environ);env.pop('ASCEND_RT_VISIBLE_DEVICES',None)"
    if runner.count(before)!=1:raise ValueError('supervisor environment binding changed')
    runner=runner.replace(before,before+"\n env['PYTHONPATH']=str(r/'source')+':'+str(r/'source/src')+':'+str(r/'source/tools/jev')")
    (output/'run.py').write_text(runner)
    result=dict(status='integrated_prepared_only',remote_root=remote,image=receipt['image'],
        parent_sha256=c['parent_sha256'],config_provenance=provenance,dual_runtime_identity=identity,
        training_dispatch_allowed=False,finite_npu_dispatch_allowed=False,
        files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()},
        pending=['fixed-image integrated source import/check','real audio batch acceptance','NPU DP4 prefix and new-process replay',
                 'full TRAIN saturation coverage','end-to-end throughput and release audit'])
    (output/'PREPARED.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();c,provenance=load_config(a.config,a.local);result=prepare(c,provenance)
    print(json.dumps(dict(status=result['status'],files=len(result['files']),training_dispatch_allowed=False),indent=2))
