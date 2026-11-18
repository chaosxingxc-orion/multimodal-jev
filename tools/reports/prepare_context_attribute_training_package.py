"""Seal a bounded attribute runtime and bind it to complete published inputs.

Local preparation only. No SSH, container creation, leases or model forward.
Runtime sealing can precede full TRAIN completion; executable package cannot.
"""
import argparse
import json
from pathlib import Path
import re
import shutil

from multimodal_jev.baseline import file_hash
from tools.reports.context_attribute_execution_candidate import authenticate,configurations,validate_pair,PREFIX
from tools.reports.context_attribute_training_candidate import derive,load_dataset
from tools.reports.context_attribute_training_contract import load_attribute_schedule
from tools.jev.prepare_context_stage2_source import verify as verify_initial

IMAGE='sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def inventory(root,exclude=()):
    root=Path(root);files={}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():raise ValueError('No symlinks in sealed bundle')
        if p.is_file() and str(p.relative_to(root)) not in exclude:
            files[str(p.relative_to(root))]=file_hash(p)
    return files


def seal_runtime(source_root,output):
    source_root=Path(source_root).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('Preserve existing runtime bundle')
    authenticate(source_root);_,derived=derive(source_root)
    paths=[]
    for directory in ('src','tools/jev','tools/reports'):
        paths.extend(p for p in (source_root/directory).rglob('*.py') if '__pycache__' not in p.parts)
    paths.append(source_root/'configs/experiments/context-native-attribute-comparison-design-v1.json')
    output.mkdir(parents=True)
    for p in paths:
        if p.is_symlink():raise ValueError('Source symlink forbidden')
        dest=output/p.relative_to(source_root);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(p,dest)
    # Ensure this repo's tools namespace cannot be shadowed by installed tools.
    for name in ('tools/__init__.py','tools/jev/__init__.py','tools/reports/__init__.py'):
        if not (output/name).exists():(output/name).write_text('')
    authenticate(output)
    if derive(output)[1]!=derived:raise ValueError('Copied trainer derivation differs')
    result=dict(schema='context-attribute-runtime-source-v1',image=IMAGE,
        training_dispatch_allowed=False,derived_trainer=derived,files=inventory(output))
    write(output/'RUNTIME.json',result)
    return result


def verify_tree(root,manifest_name,digest):
    root=Path(root)
    if root.is_symlink() or file_hash(root/manifest_name)!=digest:
        raise ValueError('Exact sealed manifest required')
    m=json.loads((root/manifest_name).read_text())
    if inventory(root,(manifest_name,))!=m['files']:
        raise ValueError('Sealed file population or bytes changed')
    return m


def command(root,c,publication_mounts):
    if (not re.fullmatch(re.escape(PREFIX)+r'context-attribute-[a-z0-9-]+',root)
            or c['physical_devices']!=[4,5,6,7] or c['start_update']!=0 or c['stop_update']!=2
            or c['attribute_source_root']!=root+'/source' or c['training_dispatch_allowed'] is not False):
        raise ValueError('Isolated bounded deployment required')
    argv=['docker','create','--pull=never','--network=none','--name',Path(root).name,
        '--cpus=24','--memory=128g','--shm-size=8g','--log-driver=json-file',
        '--log-opt','max-size=20m','--log-opt','max-file=2']
    for device in ('davinci4','davinci5','davinci6','davinci7','davinci_manager','devmm_svm','hisi_hdc'):
        argv+=['--device','/dev/'+device+':/dev/'+device]
    mounts=[root+':'+root,root+'/exchange:/jev-exchange',root+'/home:/root',root+'/cache:/jev-tmp',
        '/mnt/cephfs/xingchao:/mnt/cephfs/xingchao:ro','/usr/local/Ascend/driver:/usr/local/Ascend/driver:ro',
        c['audio_mirror_root']+':'+c['audio_mirror_root']+':ro']
    for name in ('source','initial','dataset','original-schedule','attribute-inputs','config.json','service-config.json','PREPARED.json'):
        mounts.append(root+'/'+name+':'+root+'/'+name+':ro')
    for mount in publication_mounts:
        parts=mount.split(':')
        if len(parts)!=3 or parts[2]!='ro' or not parts[0].startswith('/mnt/cephfs/xingchao/'):
            raise ValueError('Read-only publication dependency required')
        mounts.append(mount)
    for mount in mounts:argv+=['-v',mount]
    env=dict(PYTHONPATH=root+'/source:'+root+'/source/src:'+root+'/source/tools/jev',
        PYTHONDONTWRITEBYTECODE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',HCCL_DETERMINISTIC='true',TMPDIR='/jev-tmp')
    for key in ('HF_HOME','XDG_CACHE_HOME','TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR','ASCEND_WORK_PATH','ASCEND_CACHE_PATH'):
        env[key]='/jev-tmp'
    for k,v in env.items():argv+=['-e',k+'='+v]
    return argv+['--entrypoint','/opt/multimodal-jev/runtime-entrypoint.sh',IMAGE,'python',
        root+'/source/tools/reports/context_attribute_execution_candidate.py','--package',root]


def prepare(*,runtime,runtime_sha256,formal,formal_sha256,publication,publication_sha256,
            dataset,arm,algorithm,output,remote_root):
    runtime,formal,publication,output=map(Path,(runtime,formal,publication,output))
    if output.exists():raise ValueError('New package required')
    source=verify_tree(runtime,'RUNTIME.json',runtime_sha256)
    old=verify_tree(formal,'PREPARED.json',formal_sha256)
    full=verify_tree(publication,'PUBLICATION.json',publication_sha256)
    if (source['image']!=IMAGE or old['image']!=IMAGE or full['recordings']!=40513 or full['updates']!=634
            or full['training_dispatch_allowed'] is not False or arm not in full['arms']):
        raise ValueError('Complete fixed-image source and full input publication required')
    audit=json.loads((publication/'TRAIN-independent-v2.json').read_text())
    if audit['passed'] is not True or audit['recordings']!=40513 or audit['requests']!=81026:
        raise ValueError('Full actual TRAIN input evidence required')
    menu,inputs,_,_=load_dataset(dataset)
    base=json.loads((formal/'config.json').read_text());service=json.loads((formal/'service-config.json').read_text())
    verify_initial(formal/'initial',file_hash(formal/'initial/PACKAGE.json'))
    load_attribute_schedule(publication/arm,full['arms'][arm]['manifest_sha256'],
        original_schedule=formal/'source/schedule',population={r['id']:r for r in inputs},
        menu=menu,arm=arm,source_root=runtime)
    from tools.jev.context_release_mounts import publication_mounts
    mounts=publication_mounts()
    base.update(source_policy=remote_root+'/initial/policy',initial_adapter=remote_root+'/initial/adapter',
        reference_adapter=remote_root+'/initial/adapter',schedule=remote_root+'/original-schedule')
    service['policy'].update(snapshot=base['source_policy'],adapter=base['initial_adapter'])
    c,s=configurations(base,service,root=remote_root,source_root=remote_root+'/source',
        dataset=remote_root+'/dataset',schedule=remote_root+'/attribute-inputs/'+arm,
        schedule_sha256=full['arms'][arm]['manifest_sha256'],arm=arm,algorithm=algorithm,pins={})
    output.mkdir(parents=True)
    for src,name in [(runtime,'source'),(formal/'initial','initial'),(Path(dataset),'dataset'),
                     (formal/'source/schedule','original-schedule'),(publication,'attribute-inputs')]:
        shutil.copytree(src,output/name)
    external={p:h for p,h in json.loads((formal/'config.json').read_text())['pins'].items()
        if p.startswith(c['model']+'/') or p==c['tasks'] or p==c['audio_mirror_root']+'/MANIFEST.json'}
    pins={**external,**{remote_root+'/'+n:h for n,h in inventory(output).items()}}
    c['pins']=pins;s['pins']=dict(pins);validate_pair(c,s)
    write(output/'config.json',c);write(output/'service-config.json',s)
    result=dict(schema='context-attribute-bounded-training-package-v1',status='sealed_not_launched',
        training_dispatch_allowed=False,image=IMAGE,remote_root=remote_root,arm=arm,algorithm=algorithm,
        runtime_sha256=runtime_sha256,source_formal_sha256=formal_sha256,full_inputs_sha256=publication_sha256,
        files=inventory(output),argv=command(remote_root,c,mounts),
        prerequisites=['existing simple training and transfer queue completed','same exclusive DP4 lease and idle cards',
            'fixed-image runtime configuration validation','actual compact HF/vLLM probability parity',
            'independent two-update and fresh-process update1-to2 audit'],
        scope='Initial two-update candidate only. No launch, resume package or formal complex-task training admission.')
    write(output/'PREPARED.json',result)
    verify_tree(output,'PREPARED.json',file_hash(output/'PREPARED.json'))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    seal=sub.add_parser('seal-runtime');seal.add_argument('--source-root',required=True);seal.add_argument('--output',required=True)
    prep=sub.add_parser('prepare')
    for k in ('runtime','runtime-sha256','formal','formal-sha256','publication','publication-sha256',
              'dataset','arm','algorithm','output','remote-root'):prep.add_argument('--'+k,required=True)
    a=vars(p.parse_args());action=a.pop('action');r=seal_runtime(**a) if action=='seal-runtime' else prepare(**a)
    print(json.dumps({k:v for k,v in r.items() if k not in ('files','argv')},indent=2))
