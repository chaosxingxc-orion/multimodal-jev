"""Seal only missing source-policy predictions for a checkpoint curve."""
import argparse
from pathlib import Path
import json
import shutil
import tarfile
import torch
from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.source_snapshot import freeze
from multimodal_jev.local_config import load_config
from tools.reports.build_tpo_view import verified

def read(path, sha):
    return json.loads(verified(Path(path),sha).read_bytes())

def run(c):
    pkg=Path(c['output']); entry=pkg/'entry'; entry.mkdir(parents=True)
    prior=Path(c['prior_package'])
    ctl=read(prior/'operations-local.json',c['operations_sha'])['macros']['NATIVE_SEAL_OPERATIONS_CONFIG']
    def template(name):
        fn=name+'.json'
        return read(prior/'entry'/fn,ctl['entry_pins'][fn])['macros']['NATIVE_SEAL_CONTAINER_CONFIG']
    spec=template('block-plan'); targets=Path(c['targets'])
    tm=read(targets/'MANIFEST.json',spec['binding']['target_sha'])
    rows=read(targets/'rows.json',tm['files']['rows.json'])
    q=torch.load(verified(targets/'targets.pt',tm['files']['targets.pt']),weights_only=True,map_location='cpu')
    ref=Path(c['reference']); rm=read(ref/'MANIFEST.json',tm['reference_sha256'])
    records=read(ref/'records.json',rm['files']['records.json'])
    byid=read(ref/'population.json',rm['files']['population.json'])
    ids=tm['candidate_ids']; assert ids==rm['identity']['candidate_ids']
    torch.set_num_threads(c['cpu_threads'])
    order=sorted(rows,key=lambda k:rows[k]['index'])
    gold=torch.tensor([rows[k]['gold'] for k in order])
    flipped=(q.gather(-1,gold[None,:,None].expand(q.size(0),-1,1)).squeeze(-1)>
             q.masked_fill(torch.nn.functional.one_hot(gold,len(ids))[None].bool(),-torch.inf).amax(-1)).sum(0).tolist()
    roster=[];lens={};truth={}
    for j,uid in enumerate(order):
        row=rows[uid]; assert row['index']==j and byid[uid]==row['request_key']
        rec=records[row['request_key']];assert rec['policy_id']==tm['source_policy']
        source=rec['request_identity']['inputs']
        lens[uid]=dict(tokens=source['input_ids']['shape'][-1],frames=source['input_features']['shape'][-1])
        roster.append(dict(id=uid,source_group=row['group'],intent=ids[row['gold']],audio_sha256=row['audio_sha256']))
        truth[uid]=dict(row,prob=rec['prob'],projected=flipped[j])
    for name,value in [('train-roster',roster),('train-lengths',lens),('train-truth',truth)]:
        commit_json(entry/(name+'.json'),value)
    def pin(name):
        return dict(path='/jev-entry/'+name+'.json',sha256=file_hash(entry/(name+'.json')))
    tr,sl=c['train_panel'],c['selection_panel']; mid=c['checkpoint'];arm=next(iter(mid))
    assert tm['intermediate_steps']==[int(arm.removeprefix('step-'))]
    train=dict(template('calibration-0')['endpoint'],train_view=dict(roster=pin('train-roster'),
        lengths=pin('train-lengths'),recordings=len(rows),absent_classes=sorted(set(ids)-{r['intent'] for r in roster})))
    repo=Path(__file__).resolve().parents[2]
    src=freeze(repo,entry/'source',patterns=c['source_patterns'],io_workers=spec['io_workers'])
    spec.update(source=dict(root='/jev-entry/source',sha256=src['manifest_sha256']),
                endpoints={tr:train,sl:spec['endpoints'][sl]},checkpoints=mid,
                inferred_arms={tr:[arm,'tpo'],sl:[arm]},arms=[arm,'tpo'])
    spec['queue']=dict(spec['queue'],partitions={tr:len(c['devices']),sl:spec['queue']['partitions'][sl]},
                       namespace=c['namespace'])
    score=template('score'); oldroot=Path(ctl['remote_root'])
    prior_ops=Path(ctl['operations'])
    summary=read(prior_ops/'score-collected.json',c['score_sha'])
    oldplan=read(prior_ops/'block-plan-collected.json',c['prior_plan_receipt_sha'])
    score_spec=dict(spec,phase='tpo_curve_score',result_path='/jev-work/curve.json',
        curve=dict(train=tr,selection=sl,mid=arm,labels=ids,truth=pin('train-truth'),
                   old_queue=str(oldroot/'sample-blocks'),old_namespace=score['queue']['namespace'],
                   old_plan=oldplan['plan_sha256'],old_metrics=summary['datasets'][sl]['arms'],
                   old_shards=score['reused_predictions'][sl]['stage1']['shards']))
    def write(name,cfg):
        commit_json(entry/(name+'.json'),dict(schema='jev-local-macros-v1',macros={'NATIVE_SEAL_CONTAINER_CONFIG':cfg}))
    write('block-plan',spec);write('score',score_spec)
    groups=[[dict(ctl['chains'][0][-3][0])],[],[dict(ctl['chains'][0][-1][0],collect='curve.json')]]
    for i,job in enumerate(ctl['chains'][0][-2]):
        assert job['devices']==[c['devices'][i]]
        name=job['name']; worker=template(name)
        write(name,dict(spec,**{k:worker[k] for k in ['phase','output','devices','result_path']}))
        groups[1].append(job)
    shutil.copyfile(repo/'configs/portable/operations/native-seal-container.json',entry/'native-seal-container.json')
    tar=pkg/'entry.tar'
    with tarfile.open(tar,'x') as out:out.add(entry,arcname='entry')
    new={k:ctl[k] for k in ('host','lease','entrypoint','image','driver_root','shared_root','transfer_timeout','wait_for_lease','device_group_lease')}
    new.update(c['deployment'],archive=str(tar),entry_archive_sha256=file_hash(tar),chains=[groups],
               read_only_mounts=[*ctl['read_only_mounts'],f'{oldroot}:{oldroot}'],
               entry_pins={p.name:file_hash(p) for p in entry.glob('*.json')})
    commit_json(pkg/'operations-local.json',dict(schema='jev-local-macros-v1',macros={'NATIVE_SEAL_OPERATIONS_CONFIG':new}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config');p.add_argument('local')
    a=p.parse_args();run(load_config(a.config,a.local)[0])
