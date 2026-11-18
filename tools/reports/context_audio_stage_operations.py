"""Versioned actual-model stage diagnostic using the shared finite launcher."""
import argparse
import json
from pathlib import Path
import shutil
from types import FunctionType
from multimodal_jev.local_config import load_config
from tools.reports import context_acoustic_numeric_operations as parent
from tools.reports.prepare_context_batch_probe_candidate import sha, IMAGE
from tools.reports.run_context_batch_probe_candidate import verify
from tools.reports.launch_context_batch_probe_candidate import write_new

ADDITIONS=['context_audio_stage_probe.py','context_audio_stage_worker.py',
           'context_audio_stage_capture.py','context_acoustic_numeric_probe.py',
           'measure_context_acoustic_candidate.py']


def argv(c):
    args=parent.argv(c)
    key=c['remote_root']+'/package/tools/reports/context_acoustic_numeric_probe.py'
    args[args.index(key)]=c['remote_root']+'/package/tools/reports/context_audio_stage_probe.py'
    return args


def validate(c):
    parent.bindings(c)
    base=Path(c['prepared']);p=base/'package';r=json.loads((base/'PREPARED.json').read_text())
    config=json.loads((p/'config.json').read_text())
    if (r['files_sha256']!=verify(p) or r['image']!=IMAGE or r['argv']!=argv(c)
        or config['policy']['physical_devices']!=[7]
        or config['policy']['snapshot']!=c['remote_root']+'/package/policy'
        or config['policy']['adapter']!=c['remote_root']+'/package/adapter'
        or config['cache_root']!=c['cache_root'] or config['model']!=c['model']):
        raise ValueError('exact sealed stage diagnostic required')
    for name in ADDITIONS:
        if sha(p/'tools/reports'/name)!=sha(Path(c['source'])/'tools/reports'/name):
            raise ValueError('stage diagnostic source changed')
    return r,config


def prepare(c,provenance):
    parent.bindings(c)
    inherited=Path(c['parent_package'])
    if verify(inherited)!=c['parent_manifest_sha256']:raise ValueError('frozen parent changed')
    dest=Path(c['prepared']);dest.mkdir(exist_ok=False);p=dest/'package'
    shutil.copytree(inherited,p);(p/'FILES.json').unlink()
    if (p/'config.json').exists():raise ValueError('unambiguous inherited config required')
    for name in ADDITIONS:shutil.copy2(Path(c['source'])/'tools/reports'/name,p/'tools/reports'/name)
    receipts=Path(c['cache_receipts'])
    if sha(receipts)!=c['cache_receipts_sha256']:raise ValueError('cache receipts changed')
    rows=json.loads(receipts.read_text());selected=[]
    selection='first lexical ID for declared input lengths; no gold'
    if 'rollout_replay' in c:
        spec=c['rollout_replay'];path=Path(spec['journal'])
        if sha(path)!=spec['journal_sha256']:raise ValueError('original rollout journal changed')
        journal=json.loads(path.read_text())
        jobs=[j for j in journal['jobs'] if j['rollout_id']==spec['rollout_id']]
        if len(jobs)!=1:raise ValueError('one recorded rollout required')
        by_tensors={json.dumps(r['tensors'],sort_keys=True):r for r in rows}
        selected=[by_tensors[json.dumps(journal['inputs'][i],sort_keys=True)] for i in jobs[0]['indices']]
        # Original service stably sorts each whole rollout by expanded text length.
        selected=sorted(selected,key=lambda r:r['tensors']['input_ids']['shape'][1])
        if len(selected)!=32 or len({r['id'] for r in selected})!=32:
            raise ValueError('complete unique service32 batch required')
        selection=dict(mode='exact_recorded_rollout_service_order',**spec)
    else:
        for length in c['padded_frames']:
            candidates=[r for r in rows if r['tensors']['input_features']['shape']==[1,128,length]]
            selected.append(sorted(candidates,key=lambda r:r['id'])[0])
    policy=json.loads((p/'service-0.json').read_text())['policy']
    policy.update(snapshot=c['remote_root']+'/package/policy',adapter=c['remote_root']+'/package/adapter',physical_devices=[7])
    policy.pop('process_device_scope',None)
    config=dict(model=c['model'],cache_root=c['cache_root'],rows=selected,policy=policy,
        input_receipts_sha256=sha(receipts),selection=selection)
    write_new(p/'config.json',config)
    write_new(p/'FILES.json',{str(x.relative_to(p)):sha(x) for x in p.rglob('*') if x.is_file()})
    receipt=dict(status='sealed_actual_model_stage_probe',image=IMAGE,argv=argv(c),files_sha256=verify(p),
        parent_manifest_sha256=c['parent_manifest_sha256'],provenance=provenance,training_dispatch_allowed=False)
    write_new(dest/'PREPARED.json',receipt);write_new(dest/'operations.json',c)
    return validate(c)[0]


def audit(c,collection,handle):
    # Reuse transport/runtime checks; the report retains explicit non-admission.
    fn=FunctionType(parent.audit.__code__,dict(parent.audit.__globals__,validate=validate),parent.audit.__name__)
    result=fn(c,collection,handle)
    from tools.reports.context_audio_stage_probe import compare
    root=Path(collection)/'evaluation'
    actual=compare(root,validate(c)[1])
    if actual!=json.loads((root/'report.json').read_text()):raise ValueError('independent tensor comparison differs')
    return dict(result,status='stage_tensors_independently_compared_not_admission')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--local',required=True)
    p.add_argument('--action',choices=['prepare','validate','launch'],required=True);a=p.parse_args()
    c,provenance=load_config(a.config,a.local)
    ns=parent.namespace();ns.update(validate=validate,audit=audit)
    result=prepare(c,provenance) if a.action=='prepare' else validate(c)[0] if a.action=='validate' else ns['launch'](c,provenance)
    print(json.dumps(result,indent=2))
