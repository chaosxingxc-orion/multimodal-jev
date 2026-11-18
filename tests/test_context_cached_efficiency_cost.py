import copy
import json

import pytest

from tools.reports import context_cached_efficiency_cost as module
from test_context_macro_efficiency_summary import fixture as step_fixture


def fixture(tmp_path):
    run=tmp_path/'run';step_fixture(run)
    # Extend known complete two-rank records to the registered 100-step window.
    ack=json.loads((run/'activation-0001.json').read_text())
    row=json.loads((run/'rank-0/update-0001.json').read_text())
    for i in range(1,101):
        a=copy.deepcopy(ack);a['policy_id']=str(i);a['receipt']['policy_id']=str(i)
        for r in a['receipt']['replica_receipts']:r['policy_id']=str(i)
        (run/f'activation-{i:04d}.json').write_text(json.dumps(a))
        for rank in range(2):
            r=copy.deepcopy(row);r.update(update=i,next_policy_id=str(i),total_seconds=10.+rank)
            r['rollout'].update(update=i-1,reference_misses=0)
            (run/f'rank-{rank}/update-{i:04d}.json').write_text(json.dumps(r))
    (run/'result.json').write_text(json.dumps(dict(passed=True,start_update=0,stop_update=100)))
    stats=module.summarize(run,start_update=0,stop_update=100,warmup_updates=5)
    population={str(i):str(i) for i in range(40513)}
    identity=dict(policy_id='stage1',runtime_identity={'fixture':'runtime'})
    training=dict(experiment_kind='efficiency',start_update=0,stop_update=100,
        source_policy_id='stage1',base_id='base',source_contract={},release_sha256='release',
        dual_runtime_identity=identity['runtime_identity'],reference_dataset=dict(root='/data/reference/dataset',
        manifest_sha256='manifest',identity=identity,population=population),batch_config=dict(training_devices=2,
        tensor_parallel=1,pipeline_parallel=1,per_dp_replica_batch=64,microbatch=8,inference_instances=2,
        inference_batch_queries=32,source_queries=40513,epochs=2,efficiency_steps=100))
    build={k:copy.deepcopy(training[k]) for k in ('source_policy_id','base_id','source_contract','release_sha256','dual_runtime_identity')}
    build['output']='/data/reference'
    result=dict(manifest_sha256='manifest',identity=identity,population=population,recordings=40513,total_seconds=500.)
    def observed(cid,start,end):
        return dict(Id=cid,Image='image',State=dict(Status='exited',Running=False,ExitCode=0,OOMKilled=False,
            StartedAt=start,FinishedAt=end),HostConfig=dict(Devices=[dict(PathOnHost='/dev/davinci'+str(i)) for i in range(4,8)]))
    ref=observed('ref','2026-11-17T10:00:00Z','2026-11-17T10:10:00Z')
    train=observed('train','2026-11-17T10:15:00Z','2026-11-17T10:35:00Z')
    return [training,build,result,train,dict(container_id='train',image='image'),ref,dict(container_id='ref',image='image'),stats]


def test_complete_cached_loop_charges_full_reference_once(tmp_path):
    args=fixture(tmp_path);p=module.compose(*args)
    assert p['loop']['all_updates']['complete_seconds']==1100
    assert p['loop']['measured']['updates']==95
    assert p['reference']['container_non_build_seconds']==100
    assert p['training']['container_non_step_seconds']==100
    assert p['combined']['serial_container_seconds']==1800
    assert p['combined']['allocated_device_hours']==2
    assert p['combined']['training_exposures_per_container_second']==12800/1800
    assert p['combined']['gap_between_containers_seconds']==300
    assert p['reference_cost_charged_once'] and not p['matched_speedup_claim']
    assert p['correctness_audit_required'] and not p['efficacy_claim']


@pytest.mark.parametrize('fault',['partial','warmup','miss','policy','manifest','population','path',
    'failed','oom','running','image','container','overlap','timezone','reverse','loop','build','nan','cards','devices'])
def test_misleading_or_inconsistent_cost_evidence_rejected(tmp_path,fault):
    args=fixture(tmp_path);t,b,r,ti,th,ri,rh,s=args
    if fault=='partial':t['stop_update']=99
    elif fault=='warmup':s['warmup_updates']=4
    elif fault=='miss':s['all_updates']['reference_misses']=1
    elif fault=='policy':b['source_policy_id']='other'
    elif fault=='manifest':r['manifest_sha256']='other'
    elif fault=='population':r['recordings']=40512
    elif fault=='path':b['output']='/wrong'
    elif fault=='failed':ri['State']['ExitCode']=1
    elif fault=='oom':ri['State']['OOMKilled']=True
    elif fault=='running':ti['State']['Running']=True
    elif fault=='image':rh['image']='other'
    elif fault=='container':ri['Id']='other'
    elif fault=='overlap':ri['State']['FinishedAt']='2026-11-17T10:20:00Z'
    elif fault=='timezone':ri['State']['StartedAt']='2026-11-17T10:00:00'
    elif fault=='reverse':ri['State']['FinishedAt']='2026-11-17T09:00:00Z'
    elif fault=='loop':s['all_updates']['complete_seconds']=1201
    elif fault=='build':r['total_seconds']=601
    elif fault=='nan':r['total_seconds']=float('nan')
    elif fault=='cards':ri['HostConfig']['Devices'].pop()
    else:ri['HostConfig']['Devices'][0]['PathOnHost']='/dev/davinci0'
    with pytest.raises(ValueError):module.compose(*args)


@pytest.mark.parametrize('fault',[None,'hash','training-config','reference-config','reference-result','step','state','transport'])
def test_public_report_binds_measurements_to_container_transport(tmp_path,monkeypatch,fault):
    from multimodal_jev import local_checkpoint
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda _:None)
    t,b,r,ti,th,ri,rh,stats=fixture(tmp_path)
    th['remote_root']='/data/train';rh['remote_root']='/data/ref'
    values=dict(training_config=t,reference_build_config=b,reference_result=r,
        training_inspect=ti,training_handle=th,reference_inspect=ri,reference_handle=rh)
    c=dict(run_root=str(tmp_path/'run'),output=str(tmp_path/'cost.json'))
    def bind(key,value):
        path=tmp_path/(key+'.json');path.write_text(json.dumps(value))
        c[key]=dict(path=str(path),sha256=module.file_hash(path))
    for key in ('training_config','reference_build_config','reference_result'):bind(key,values[key])
    th['config_sha256']=c['training_config']['sha256'];rh['config_sha256']='reference-config-hash'
    for key in ('training_handle','reference_handle','training_inspect','reference_inspect'):bind(key,values[key])
    def transport(handle,observed,files):
        return dict(status='transport_complete_independent_audit_required',container_id=handle['container_id'],
            image=handle['image'],remote_root=handle['remote_root'],state=copy.deepcopy(observed['State']),
            bad_pins=[],bad_package=[],launch_binding_mismatches=[],files=files)
    tt=transport(th,ti,dict({'run/'+k:v for k,v in stats['files'].items()},**{'config.json':th['config_sha256']}))
    rt=transport(rh,ri,{'config.json':rh['config_sha256'],
        'reference-build-config.json':c['reference_build_config']['sha256'],
        'reference-build/result.json':c['reference_result']['sha256']})
    if fault=='training-config':tt['files']['config.json']='other'
    elif fault=='reference-config':rt['files']['reference-build-config.json']='other'
    elif fault=='reference-result':rt['files']['reference-build/result.json']='other'
    elif fault=='step':tt['files']['run/rank-0/update-0001.json']='other'
    elif fault=='state':tt['state']['ExitCode']=7
    elif fault=='transport':rt['container_id']='other'
    bind('training_transport',tt);bind('reference_transport',rt)
    if fault=='hash':c['training_transport']['sha256']='other'
    if fault:
        with pytest.raises(ValueError):module.run(c,{})
        assert not (tmp_path/'cost.json').exists()
    else:
        proof=module.run(c,{})
        assert proof['combined']['serial_container_seconds']==1800
        assert proof['correctness_audit_required']
        assert json.loads((tmp_path/'cost.json').read_text())==proof
