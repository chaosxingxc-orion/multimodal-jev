"""Seal and compare two concurrent Stage1 acoustic paths on matched devices and TRAIN draws."""
from copy import deepcopy
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile

from multimodal_jev.baseline import file_hash
from multimodal_jev.context_topology_batch_plan import derive
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.source_snapshot import freeze
from multimodal_jev.stage1_mixed_schedule import schedule
from multimodal_jev.stage1_batch_probe import select_batches
from tools.reports.native_seal_operations import run


def read(path):
    return json.loads(path.read_bytes())


def execute(c):
    base=Path(c['package'])
    old=read(base/'operations-local.json')['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']
    out=Path(c['output']);entry=out/'entry'
    parent=Path(old['remote_root'])
    def remote(*args):
        return subprocess.check_output(['ssh','-o','BatchMode=yes',old['host'],shlex.join(args)])
    formal=json.loads(remote('cat',str(parent/'formal/config.json')))
    if (file_hash(base/'entry.tar')!=old['entry_archive_sha256'] or
            remote('sha256sum',str(parent/'formal/config.json')).decode().split()[0]!=c['formal_sha256']):
        raise ValueError('unauthenticated predecessor')
    entry.mkdir(parents=True,exist_ok=False)
    for p in (base/'entry').glob('*.json'):
        if file_hash(p)!=old['entry_pins'][p.name]:
            raise ValueError('parent entry changed')
        shutil.copyfile(p,entry/p.name)
    repo=Path(__file__).resolve().parents[2]
    source=freeze(repo,entry/'source',patterns=c['source_patterns'],io_workers=c['io_workers'])
    real=[json.loads(x) for x in (Path(formal['release'])/'learning.jsonl').read_text().splitlines()]
    pop={r['id']:r for r in real+read(entry/'synthetic.json')}
    jobs=[]
    for arm,devices in c['devices'].items():
        cfg=deepcopy(formal)
        cfg['batch_config']['training_devices']=len(devices)
        plan=derive(cfg['batch_config']);width=plan['global_batch']
        batches,manifest=schedule(pop,cfg['batch_config'],cfg['mixed']['seed'])
        spans=[dict(s,steps=s['recordings']//width) for s in c['spans']]
        count=sum(s['steps'] for s in spans)
        if not any(slot['id']==c['pressure_anchor'] for s in spans if s['name']=='long_audio'
                   for b in batches[s['offset']//width:(s['offset']+s['recordings'])//width] for slot in b['slots']):
            raise ValueError('registered long recording absent')
        select_batches(batches,width,dict(spans=spans),count)
        path=entry/'comparison-schedule.json'
        if not path.exists():commit_json(path,manifest)
        elif read(path)!=manifest:raise ValueError('unmatched schedules')
        cfg['mixed']['schedule_manifest']='/jev-entry/'+path.name
        cfg.update(physical_devices=devices,global_batch=width,schedule_sha256=file_hash(path),
             engineering_prefix=True,training_dispatch_allowed=False,start_update=0,stop_update=count,
             parity_passes=0,evaluation_steps=[],reference_endpoint=plan['total_updates'],checkpoint_interval=count,
             batch_probe=dict(spans=spans),tower_witness=True)
        if arm=='candidate':cfg['training_acoustic_engine']=c['engine']
        path=entry/f'{arm}-config.json';commit_json(path,cfg)
        leaf=dict(phase='card_stage1',role=arm,work='/jev-work',source=dict(root='/jev-entry/source',sha256=source['manifest_sha256']),
                  io_workers=c['io_workers'],master_port=c['master_port'],batch_probe=dict(path='/jev-entry/'+path.name,sha256=file_hash(path)))
        (entry/f'{arm}.json').write_text(json.dumps(dict(schema='jev-local-macros-v1',macros=dict(NATIVE_SEAL_CONTAINER_CONFIG=leaf))))
        jobs.append(dict(name=arm,devices=devices,collect=arm+'/receipt.json',**c['resources']))
    archive=out/'entry.tar'
    with tarfile.open(archive,'x') as tar:tar.add(entry,arcname='entry')
    tag=c['tag']
    cfg=dict(old,archive=str(archive),entry_archive_sha256=file_hash(archive),tag=tag,
             remote_root=str(parent.with_name(tag)),operations=str(out/'ops'),chains=[[jobs]],
             read_only_mounts=[*old['read_only_mounts'],str(parent/'prepare')+':/jev-work/prepare'],
             entry_pins={str(p.relative_to(entry)):file_hash(p) for p in [*entry.glob('*.json'),entry/'source/FILES.json']})
    cfg.pop('idle_devices',None)
    commit_json(out/'operations.json',cfg)
    if run(cfg):raise RuntimeError('paired probe failed')
    result=audit(c,cfg)
    commit_json(out/'comparison.json',result)
    return result


def audit(c,cfg):
    from statistics import mean
    root=Path(cfg['remote_root'])
    ops=Path(cfg['operations'])
    arms={}
    for arm,devices in c['devices'].items():
        receipt=read(ops/f'{arm}-collected.json')
        if receipt['outcome']!='completed' or receipt['trainer_exit_code']!=0:
            raise ValueError('incomplete arm')
        reread=json.loads(subprocess.check_output(['ssh',cfg['host'],shlex.join(['cat',str(root/arm/'receipt.json')])]))
        if receipt!=reread:raise ValueError('collected arm changed')
        steps=[s for span in receipt['spans'] for s in span['updates']]
        program="""import json,sys,pathlib
p=pathlib.Path(sys.argv[1]); print(json.dumps([[json.loads((p/f'rank-{r}'/f'tower-{i:04d}.json').read_bytes()) for r in range(int(sys.argv[2]))] for i in range(1,int(sys.argv[3])+1)]))"""
        records=json.loads(subprocess.check_output(['ssh',cfg['host'],shlex.join(['python3','-c',program,str(root/arm/'run'),str(len(devices)),str(len(steps))])]))
        arms[arm]=(receipt,records)
    old,new=[arms[k] for k in ('baseline','candidate')]
    numerical=True;error=0.
    for x,y in zip(old[1],new[1],strict=True):
        for a,b in zip(x,y,strict=True):
            error=max(error,max(abs(i-j) for i,j in zip(a['logits'],b['logits'],strict=True)))
            numerical &= all(a[k]==b[k] for k in ('gradient','parameters','sample_identity','sampler_source_update','raw_grad_norm','loss'))
    segments={}
    for a,b in zip(old[0]['spans'],new[0]['spans'],strict=True):
        x=[r['actor_seconds'] for r in a['updates']];y=[r['actor_seconds'] for r in b['updates']]
        wall_a=[r['seconds'] for r in a['updates']];wall_b=[r['seconds'] for r in b['updates']]
        q=math.ceil(.9*len(x))-1
        segments[a['name']]=dict(old_actor=mean(x),new_actor=mean(y),wall_ratio=sum(wall_b)/sum(wall_a),
                                 p90_ratio=sorted(wall_b)[q]/sorted(wall_a)[q])
    passed=numerical and error<=c['logit_atol'] and all(segments[k]['wall_ratio']<=c['max_time_ratio']
        and segments[k]['p90_ratio']<=1 for k in ('ordinary','long_audio'))
    return dict(passed=passed,identical_training_state=numerical,max_abs_logit_error=error,segments=segments)
