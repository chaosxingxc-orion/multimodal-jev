"""Seal actual-logits probe from authenticated existing assets; never dispatch.

All host paths come from local macros. The original package remains immutable.
The release is an inference candidate, not a trained or admitted runtime.
"""
import argparse
import json
from pathlib import Path
import shutil

from multimodal_jev.local_config import load_config
from multimodal_jev.context_logits_wire import WIRE
from tools.reports.context_dual_rollout_candidate import RUNTIME
from tools.reports.prepare_context_batch_probe_candidate import sha
from tools.reports.launch_context_batch_probe_candidate import validate
from tools.reports.run_context_batch_probe_candidate import verify
from tools.reports.run_context_dual_probe_candidate import runner_namespace

ADDITIONS=[
    'src/multimodal_jev/context_logits_wire.py',
    'tools/reports/context_dual_rollout_candidate.py',
    'tools/reports/context_dual_worker_candidate.py',
    'tools/reports/context_dual_measurement_candidate.py',
    'tools/reports/run_context_dual_probe_candidate.py']


def rebind_argv(argv,old_root,new_root,old_model,new_model,driver,device,tag):
    # Exact root prefix substitution, never broad replacement of version text.
    out=[]
    for value in argv:
        value=value.replace(old_root,new_root)
        if value==old_model+':'+old_model+':ro':value=new_model+':'+new_model+':ro'
        if value=='/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro':
            value=driver+':/usr/local/Ascend/driver:ro'
        if value=='/dev/davinci7:/dev/davinci7':value=f'/dev/davinci{device}:/dev/davinci{device}'
        if value=='context-batch-probe-npu-v1':value=tag
        if value==new_root+'/package/tools/reports/run_context_batch_probe_candidate.py':
            value=new_root+'/package/tools/reports/run_context_dual_probe_candidate.py'
        out.append(value)
    return out


def prepare(config,provenance):
    source=Path(config['source']).resolve();original=Path(config['prepared']).resolve()
    receipt,old_config=validate(original)
    output=Path(config['output']).resolve();remote=config['remote_root']
    if (not Path(remote).is_absolute() or ':' in remote or '..' in Path(remote).parts
            or remote==receipt['remote_root'] or config['tag']=='context-batch-probe-npu-v1'
            or config['device']!=7):
        raise ValueError('distinct explicit local runtime root and original device7 lease required')
    if output==original or original in output.parents:
        raise ValueError('source package must remain immutable')
    runner_namespace(source)
    output.mkdir(exist_ok=False);package=output/'package'
    shutil.copytree(original/'package',package)
    (package/'FILES.json').unlink()
    for name in ADDITIONS:
        dest=package/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/name,dest)
    policy=dict(old_config['policy'],model=config['model'],
        snapshot=remote+'/package/policy',adapter=remote+'/package/adapter',
        candidate_runtime=RUNTIME,output_protocol=WIRE,physical_devices=[config['device']])
    updated=dict(old_config,policy=policy,source_root=remote+'/package',inputs=remote+'/package/inputs.pt')
    updated['pins']={remote+'/package/'+str(p.relative_to(package)):sha(p)
        for p in package.rglob('*') if p.is_file() and p!=package/'config.json'}
    (package/'config.json').write_text(json.dumps(updated,indent=2)+'\n')
    files={str(p.relative_to(package)):sha(p) for p in package.rglob('*') if p.is_file()}
    (package/'FILES.json').write_text(json.dumps(files,indent=2)+'\n')
    identity=verify(package)
    if verify(original/'package')!=receipt['files_sha256']:
        raise ValueError('original package changed')
    result=dict(status='sealed_not_launched',image=receipt['image'],remote_root=remote,
        source_files=len(files),files_sha256=identity,parent_files_sha256=receipt['files_sha256'],
        config_provenance=provenance,host=config['host'],tag=config['tag'],
        argv=rebind_argv(receipt['argv'],receipt['remote_root'],remote,
            old_config['policy']['model'],config['model'],config['driver_root'],config['device'],config['tag']),
        prerequisites=receipt['prerequisites'],predecessor=receipt['predecessor'],
        training_dispatch_allowed=False,launch_dispatch_allowed=False,
        scope='Actual logits/prob serial-batch candidate; no NPU execution, dual sampling, training or recovery proof',
        pending=['existing queue completion and exclusive lease','macro-aware launch/collection adapter',
                 'real-audio independent audit','full TRAIN ranking saturation coverage',
                 'dual actor integration and nonzero update/resume'])
    (output/'PREPARED.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    a=p.parse_args();config,provenance=load_config(a.config,a.local)
    result=prepare(config,provenance)
    print(json.dumps({k:result[k] for k in ('status','source_files','files_sha256','training_dispatch_allowed')},indent=2))
