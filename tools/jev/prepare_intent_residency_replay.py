#!/usr/bin/env python3
"""Freeze a real-ASR offload replay against accepted D update2, no dispatch."""
import argparse
from pathlib import Path
import shutil
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.container_pins import validate_bind_pins
from queue_span_native_actor_npu import read,completed
from queue_intent_finite_npu import verify_final,verify_gate
from queue_span_dp4_resume import check_pins
from probe_intent_residency_replay_npu import validate


def prepare(root,reference,suite_reference):
    root,reference,suite_reference=map(Path,(root,reference,suite_reference))
    if root.exists():raise FileExistsError('immutable replay plan exists')
    original=read(reference/'config.json');image=read(reference/'launch.json')['image']
    for path in (reference,suite_reference):
        launch=read(path/'launch.json')
        if launch['image']!=image or not completed(launch['container_id'],image):
            raise ValueError('actual accepted reference exits required')
    verify_final(suite_reference);verified=read(suite_reference/'verified.json')
    if verified['root']!=str(reference) or verified['updates']!=2 or verified['arm']!='D':
        raise ValueError('accepted original D second update required')
    verify_gate(suite_reference,original['pilot'],original['manifest_sha256'])
    root.mkdir();repo=Path(__file__).resolve().parents[2]
    common=dict(original['pins']);common.update(verified['files'])
    for path in (reference/'config.json',reference/'launch.json',suite_reference/'config.json',suite_reference/'launch.json',suite_reference/'ACCEPTED.json',suite_reference/'verified.json'):
        common[str(path)]=file_hash(path)
    for rank in range(4):
        path=reference/'run'/f'rank-{rank}'/'optimizer-local.pt';common[str(path)]=file_hash(path)
    check_pins(common)
    for kind,template in (('training',reference),('suite',suite_reference)):
        out=root/kind;out.mkdir()
        for folder in ('src','tools'):shutil.copytree(repo/folder,out/'source'/folder,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        source={str(p.relative_to(out/'source')):file_hash(p) for p in (out/'source').rglob('*') if p.is_file()}
        atomic_json(out/'source-manifest.json',source)
        pins={**common,**{str(out/'source'/p):h for p,h in source.items()}}
        prior=read(template/'launch.json');argv=[x.replace(str(template),str(out)) for x in prior['argv']]
        argv[argv.index('--name')+1]='multimodal-jev-'+root.name+'-'+kind
        # Keep original immutable evidence sources even when reference pins
        # refer to the old training root. Mount all local evidence explicitly.
        filtered=[];i=0
        while i<len(argv):
            if argv[i]=='-v' and argv[i+1].split(':')[1].startswith('/home/'):
                i+=2;continue
            filtered.append(argv[i]);i+=1
        argv=filtered;extra=[]
        for index,(path,h) in enumerate(pins.items()):
            if not path.startswith('/home/'):continue
            dest=out/'evidence'/f'{index}-{Path(path).name}';dest.parent.mkdir(exist_ok=True);shutil.copyfile(path,dest)
            if file_hash(dest)!=h:raise ValueError('local evidence changed')
            extra+=['-v',f'{dest}:{path}:ro']
        argv[argv.index(image):argv.index(image)]=extra
        if kind=='training':
            cfg=dict(original,pins=pins,replay_reference=str(reference),replay_reference_suite=str(suite_reference));validate(cfg)
            argv=[s.replace('train_intent_finite_npu.py','probe_intent_residency_replay_npu.py') for s in argv]
        else:
            cfg={k:v for k,v in read(suite_reference/'config-template.json').items() if k!='pins'}
            cfg.update(training_root=str(root/'training'),adapter=str(root/'training/run/adapter'),adapter_id=None,pins=pins)
            argv=[s.replace('run_intent_finite_acceptance.py','run_intent_residency_acceptance.py') for s in argv]
        atomic_json(out/'config-template.json',cfg)
        atomic_json(out/'prepared.json',dict(root=str(out),image=image,physical_devices=[4,5,6,7] if kind=='training' else [4],
            create_argv=argv,config_template_sha256=file_hash(out/'config-template.json'),source_manifest_sha256=file_hash(out/'source-manifest.json'),pending_pins=[]))
        atomic_json(out/'known-bind-pins.json',validate_bind_pins(argv,pins))
    protocol=dict(schema='intent-residency-replay-v1',reference=str(reference),reference_suite=str(suite_reference),
        updates=1,replays_update=2,arm='D',world_size=4,source_requests=8,image=image,
        prerequisites='Existing queue drained and synthetic NPU FSDP residency accepted; exact card ownership before dispatch.',
        acceptance='Real offload/reload on all ranks, exact post-update FP32 master/Adam, same BF16 adapter as reference, complete178 HF/vLLM gate.',
        scope='Engineering replay only. Does not add a learning curve point or prove actor/vLLM co-residency. Prepared, not dispatched.')
    atomic_json(root/'protocol.json',protocol)
    return dict(root=str(root),protocol_sha256=file_hash(root/'protocol.json'),status='prepared_not_dispatched')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','reference','suite-reference'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(prepare(a.root,a.reference,a.suite_reference))
