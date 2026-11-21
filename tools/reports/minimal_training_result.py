"""Compact closed training results and retire explicit local process artifacts.

The legacy auditor finishes once. Its report is retained as a receipt, while
curve reconstruction uses compact scalar records. No model, dataset or NPU call.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

from multimodal_jev.local_config import load_config
from multimodal_jev.local_runtime_storage import check_paths
from tools.cache_cleanup import delete_one, identity, protected
from tools.reports.committed_signal_report import write


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while block:=f.read(4*1024*1024):h.update(block)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text())


def regular(p):
    p=Path(p)
    if not p.is_absolute() or any(q.is_symlink() for q in [p,*p.parents]) or not p.is_file():
        raise ValueError('absolute regular file required: '+str(p))
    return p


def compact(row,rank):
    # Explicit whitelist: no per-query trajectories, tensors, activations or
    # optimizer copies. Metric vectors remain intact for faithful aggregation.
    keys=('update','preparation_seconds','actor_seconds','total_seconds','metrics',
          'next_policy_id','real_queries','peak_allocated','checkpoint')
    result={k:row[k] for k in keys if k in row}
    result['rank']=rank
    result['rollout']={k:v for k,v in row['rollout'].items() if k in (
        'update','behavior_policy_id','fixed_reference_policy_id','reference_misses',
        'reference_seconds','behavior_and_credit_seconds','total_seconds',
        'bulk_reference_calls','bulk_behavior_calls','switches')}
    return result


def deletable(name):
    # Exact relative members of the sealed collector manifest only.
    return bool(re.fullmatch(r'run/(?:execution-\d+\.pt|groups-\d+\.pt|'
        r'(?:adapter|policy)-\d+/.+|rank-\d+/(?:update-\d+\.json|checkpoint-\d+\.pt)|'
        r'pool-journal/.+|batch-\d+\.json)',name))


def gate(c):
    terminal=Path(c['completion']['path'])
    if not terminal.exists():return False
    value=read(regular(terminal))
    if any(value.get(k)!=v for k,v in c['completion']['equals'].items()):
        raise ValueError('completion receipt failed; preserve raw evidence')
    owner=c['owner'];proc=Path('/proc')/str(owner['pid'])/'stat'
    try:
        stat=proc.read_text().split(') ',1)[1].split()
        if stat[19]==owner['start_ticks'] and stat[0]!='Z':return False
    except FileNotFoundError:pass
    return True


def validate(c):
    if c['schema']!='minimal-training-result-v1' or c['interval_seconds']!=900:
        raise ValueError('versioned contract and 900 second interval required')
    for name in ('collection','output'):
        p=Path(c[name])
        if not p.is_absolute() or p.resolve()!=p or len(p.parts)<4:raise ValueError('narrow absolute paths required')
    root,out=Path(c['collection']),Path(c['output'])
    if out.is_relative_to(root) or root.is_relative_to(out):raise ValueError('disjoint result bundle required')
    check_paths({'collection':root,'output':out})
    if c['max_delete_bytes']<=0 or c['max_delete_files']<=0:raise ValueError('deletion budget required')


def prepare(c):
    validate(c);out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
    marker=out/'CAPSULE.json'
    if marker.exists():
        manifest=read(marker)
        if manifest['config_sha256']!=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest():
            raise ValueError('capsule config changed')
        verify_capsule(out,manifest);return manifest
    if (out/'preparing.json').exists():raise ValueError('incomplete capsule; use new output')
    write(out/'preparing.json',{'pid':os.getpid()})
    transport=read(regular(c['transport']));root=Path(c['collection'])
    if transport['status']!='transport_complete_independent_audit_required' or transport['state']['Running'] or transport['state']['ExitCode']:
        raise ValueError('successfully closed complete transport required')
    hashes=transport['files'];pins={};files={}
    def copy(source,name,expected=None):
        source=regular(source);h=sha(source)
        if expected is not None and h!=expected:raise ValueError('source changed: '+str(source))
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        if sha(target)!=h:raise ValueError('result copy differs')
        files[name]=h;return h
    for name in ('config.json','run/result.json'):
        copy(root/name,name,hashes[name])
    for name,pin in c['retain_files'].items():
        copy(pin['path'],name,pin['sha256'])
    assets=read(regular(c['selection_assets']['path']))
    if sha(c['selection_assets']['path'])!=c['selection_assets']['sha256']:raise ValueError('selection assets report changed')
    # Selection has a sealed independent copy of every preregistered policy.
    for n,h in assets['files'].items():
        f=regular(Path(assets['root'])/n)
        if sha(f)!=h:raise ValueError('selection dependency changed')
        pins[str(f)]=h
    pins[c['selection_assets']['path']]=c['selection_assets']['sha256']
    final=c['final_step'];prefixes=(f'run/policy-{final:04d}/',f'run/adapter-{final:04d}/')
    for n,h in assets['files'].items():
        if n.startswith(prefixes):copy(Path(assets['root'])/n,'model/'+n.split('/',1)[1],h)
    if not any(n.startswith('model/policy-') for n in files):raise ValueError('final policy missing')
    records=[f'run/rank-{rank}/update-{step:04d}.json' for step in range(1,final+1) for rank in range(c['world_size'])]
    def one(n):
        f=regular(root/n)
        if sha(f)!=hashes[n]:raise ValueError('curve source changed')
        value=read(f);step=int(re.search(r'update-(\d+)',n)[1]);rank=int(re.search(r'rank-(\d+)',n)[1])
        if value['update']!=step:raise ValueError('curve step mismatch')
        return compact(value,rank)
    with ThreadPoolExecutor(max_workers=c['io_workers']) as pool:
        rows=list(pool.map(one,records))
    write(out/'training-curve.json',{'rows':rows,'rank_aggregation':'preserve_each_rank_and_metric_vector','source_sha256':{n:hashes[n] for n in records}})
    files['training-curve.json']=sha(out/'training-curve.json')
    write(out/'references.json',c['references']);files['references.json']=sha(out/'references.json')
    manifest=dict(schema='minimal-training-capsule-v1',config_sha256=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest(),
        files=files,protected_sources=pins,container_id=transport['container_id'],
        training_audit_pending=True,scope='Scalar curve and final inference reload; full trajectory replay is intentionally not retained.')
    write(marker,manifest);verify_capsule(out,manifest);return manifest


def verify_capsule(out,m):
    for n,h in m['files'].items():
        if sha(regular(Path(out)/n))!=h:raise ValueError('capsule changed: '+n)


def apply(c):
    validate(c)
    if not gate(c):return {'status':'waiting_for_audit_owner_exit','deleted_bytes':0}
    out=Path(c['output']);m=prepare(c);verify_capsule(out,m)
    if (out/'COMPLETE.json').exists():return read(out/'COMPLETE.json')
    if (out/'deleting.json').exists():raise ValueError('partial deletion retained; explicit reconciliation required')
    for p,h in m['protected_sources'].items():
        if sha(regular(p))!=h:raise ValueError('protected selection dependency changed')
    # Preserve actual audit conclusions before retiring inputs; this receipt
    # cannot later be mistaken for a repeatable full trajectory audit.
    audit_receipts={}
    for i,p in enumerate(c['audit_reports']):
        regular(p);name=f'audit-receipt-{i}.json';shutil.copyfile(p,out/name)
        audit_receipts[name]=sha(out/name)
    write(out/'AUDIT-RECEIPTS.json',dict(training_audit_pending=False,files=audit_receipts,full_trajectory_reaudit_available=False))
    root=Path(c['collection']);t=read(c['transport']);candidates=[]
    protection=[Path(p) for p in c['protected_paths']]+[out]
    paths=[(root/n,h) for n,h in t['files'].items() if deletable(n)]
    paths.append((Path(c['archive']),t['archive_sha256']))
    for path,h in paths:
        regular(path)
        if protected(path,protection):raise ValueError('candidate overlaps protected dependency')
        candidate_root=root if path.is_relative_to(root) else path.parent
        candidates.append(dict(path=str(path),root=str(candidate_root),relative=str(path.relative_to(candidate_root)),sha256=h,identity=identity(path.stat())))
    size=sum(r['identity']['size'] for r in candidates)
    if len(candidates)>c['max_delete_files'] or size>c['max_delete_bytes']:raise ValueError('deletion budget exceeded')
    write(out/'deletion-plan.json',{'files':candidates,'logical_bytes':size})
    # Authenticate once in parallel before any deletion, then delete_one
    # checks exact inode/content again immediately before unlinkat.
    with ThreadPoolExecutor(max_workers=c['io_workers']) as pool:
        actual=list(pool.map(lambda r:sha(r['path']),candidates))
    if actual!=[r['sha256'] for r in candidates]:raise ValueError('candidate content changed')
    if not gate(c):raise ValueError('owner became active')
    write(out/'deleting.json',{'files':len(candidates),'bytes':size})
    with (out/'deleted.jsonl').open('x') as log:
        for row in candidates:
            delete_one(row,protection);log.write(json.dumps({'path':row['path'],'bytes':row['identity']['size']})+'\n');log.flush()
    verify_capsule(out,m)
    result=dict(status='minimal_result_retained_process_artifacts_deleted',deleted_files=len(candidates),deleted_bytes=size,
        retained_bytes=sum((out/n).stat().st_size for n in m['files']),full_trajectory_reaudit_available=False)
    write(out/'COMPLETE.json',result);return result


def run(c,action):
    validate(c);out=Path(c['output']);out.mkdir(parents=True,exist_ok=True)
    with (out/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if action=='prepare':return prepare(c)
        if action=='apply':return apply(c)
        prepare(c)
        while not gate(c):
            write(out/'waiting.json',{'status':'waiting_for_audit_owner_exit','interval_seconds':900,'checked_unix':time.time()})
            time.sleep(c['interval_seconds'])
        return apply(c)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['prepare','apply','watch'],required=True);a=p.parse_args()
    c,_=load_config(a.config,a.local);r=run(c,a.action)
    print(json.dumps({k:v for k,v in r.items() if k not in ('files','protected_sources')}))
