#!/usr/bin/env python3
"""Generate authenticated Docker create/start commands for an existing probe.

Does not allocate cards or launch. Publish launch.json after docker create and
before docker start, so the controller never races its own receipt publication.
"""
import argparse
import json
from pathlib import Path
import shutil
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.span_pair_pilot import pilot_rows


def prepare(root):
    repo=Path(__file__).resolve().parents[2]
    parent=Path('/mnt/cephfs/xingchao/experiments/multimodal-jev')
    root=root.resolve()
    if root.parent!=parent or not root.is_dir() or (root/'launch.json').exists() or (root/'run').exists():
        raise ValueError('unlaunched exclusive prepared root required')
    ready=json.loads((root/'prepared.json').read_text())
    if ready['launched'] is not False or ready['source_manifest_sha256']!=file_hash(root/'source-manifest.json') or ready['protocol_sha256']!=file_hash(root/'protocol.json'):
        raise ValueError('prepared identity differs')
    manifest=json.loads((root/'source-manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if file_hash(root/'source'/name)!=digest:raise ValueError('prepared source changed')
    configs={name:json.loads((root/(name+'.json')).read_text()) for name in ('pair0','pair05')}
    evidence={}
    for name,c in configs.items():
        protocol,_,_,_=pilot_rows(c)
        for path,digest in protocol['pins'].items():
            original=Path(path)
            if original.is_relative_to(repo):
                relative=original.relative_to(repo)
                if relative.parts[0]!='docs':raise ValueError('only repository document evidence may be copied')
                shared=root/'evidence'/relative
                shared.parent.mkdir(parents=True,exist_ok=True)
                if shared.exists():
                    if file_hash(shared)!=digest:raise ValueError('shared evidence differs')
                else:shutil.copyfile(original,shared)
                if file_hash(shared)!=digest:raise ValueError('evidence copy differs')
                shared.chmod(0o644)
                evidence[path]=dict(source=str(shared),sha256=digest)
        if c['pair_weight']!=dict(pair0=0.,pair05=.5)[name]:raise ValueError('branch identity differs')
        for path,digest in c['pins'].items():
            if file_hash(path)!=digest:raise ValueError('branch asset changed')
    if any(c['physical_devices']!=[5] for c in configs.values()):raise ValueError('prepared card5 branches required')
    controller=root/'source/tools/jev/run_span_pair_pilot.py'
    if not controller.is_file():raise ValueError('complete acceptance controller required')
    argv=['docker','create','--pull=never','--network=none','--name','multimodal-jev-'+root.name,
          '--cpus=8','--memory=64g','--shm-size=4g']
    for dev in ('davinci5','davinci_manager','devmm_svm','hisi_hdc'):argv+=['--device',f'/dev/{dev}:/dev/{dev}']
    for env in ('PYTHONDONTWRITEBYTECODE=1','HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1',
                'TOKENIZERS_PARALLELISM=false','OMP_NUM_THREADS=4','VLLM_WORKER_MULTIPROC_METHOD=spawn',
                'PYTHONPATH='+str(root/'source/src')):argv+=['-e',env]
    # Remote host need not contain the local repo. Mount each authenticated
    # document from shared storage; avoid read-only parent/nested mount creation.
    for path,mode in [('/usr/local/Ascend/driver','ro'),('/mnt/cephfs/xingchao/models','ro'),
                      ('/mnt/cephfs/xingchao/datasets','ro'),(str(parent),'ro'),
                      (str(root),'rw'),(str(root/'source'),'ro')]:argv+=['-v',f'{path}:{path}:{mode}']
    for rel in ('src','tools','configs/tasks/slurp/temporal/tasks-v1.json'):
        argv+=['-v',f'{root}/source/{rel}:{repo}/{rel}:ro']
    for target,item in evidence.items():argv+=['-v',f'{item["source"]}:{target}:ro']
    argv += [ready['image'],'timeout','--signal=TERM','--kill-after=30s','7200s','python3','-B',str(controller),
             '--root',str(root),'--output',str(root/'run')]
    return dict(root=str(root),prepared=True,launched=False,create_argv=argv,
        start_argv_template=['docker','start','<exact-created-container-id>'],
        evidence=evidence,receipt_fields=dict(root=str(root),image=ready['image'],physical_devices=[5],
            source_manifest_sha256=file_hash(root/'source-manifest.json'),protocol_sha256=file_hash(root/'protocol.json'),
            controller_sha256=file_hash(controller),config_sha256={n:file_hash(root/(n+'.json')) for n in configs}),
        required_sequence=['Confirm physical NPU5 and its queued work are free.',
            'docker create; record exact container ID and exit status in root/launch.json.',
            'docker start that exact ID; save start status separately; never recreate on observation timeout.'],
        scope='Launch argv preparation only. Source and evidence mounts complete; no device allocation or NPU execution.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=prepare(a.root);atomic_json(a.output,r);print(json.dumps({k:v for k,v in r.items() if k!='create_argv'},indent=2))
