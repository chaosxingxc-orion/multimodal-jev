"""CPU topology/transport tests; surrogate probabilities are not NPU evidence."""
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tools.reports import context_multidevice_evaluation as topology
from tools.reports import prepare_multidevice_selection as package
from tools.reports import multidevice_selection_operations as operations
from test_prepare_context_replica_selection import setup,write


@pytest.mark.parametrize('devices',[[7],[7,4],[7,4,6,5]])
def test_derived_topology_and_real_hf_child(tmp_path,monkeypatch,devices):
    from multimodal_jev import local_runtime_storage,local_checkpoint
    from tools.jev.context_process_supervisor import stop_owned
    monkeypatch.setattr(local_runtime_storage,'require_training_storage',lambda _:None)
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda _:None)
    cfg=dict(devices=devices,service_batch_queries=32,waves_per_macro=1)
    t=topology.topology(cfg);m=dict(t,evaluation_topology=cfg)
    assert t['macro_queries']==len(devices)*32
    write(tmp_path/'parity-hf.json',dict(expected_population=9226,source_step=318,
        process_device_scope={'visible_ordinal':t['hf_visible_ordinal']}))
    for s in m['inference_services']:write(tmp_path/f"service-{s['name']}.json",{'socket_path':str(tmp_path/s['name'])})
    (tmp_path/'selection-client.py').write_text('import json,os\nfrom pathlib import Path\nr=Path(__file__).parent\n'
        +f"assert os.environ['ASCEND_RT_VISIBLE_DEVICES']=={t['hf_visible_ordinal']!r}\n"
        +"(r/'hf').mkdir()\n(r/'hf/terminal.json').write_text('{\"exit_code\":0}')\n")
    calls=[]
    def supervise(**kw):
        record=json.loads((tmp_path/'hf-process/terminal.json').read_text())
        with pytest.raises(ProcessLookupError):os.kill(record['pid'],0)
        calls.append(kw);return {'passed':True}
    monkeypatch.setitem(sys.modules,'context_process_supervisor',SimpleNamespace(supervise_services=supervise,stop_owned=stop_owned))
    (tmp_path/'cache').mkdir()
    script=topology.runner(m).replace("Path('/jev-tmp')",'Path('+repr(str(tmp_path/'cache'))+')')
    with pytest.raises(SystemExit) as e:exec(compile(script,'<runner>','exec'),{'__file__':str(tmp_path/'run.py')})
    assert e.value.code==0 and len(calls)==1
    assert [s['env']['ASCEND_RT_VISIBLE_DEVICES'] for s in calls[0]['services']]==[str(sorted(devices).index(d)) for d in devices]
    assert calls[0]['actor_env']['ASCEND_RT_VISIBLE_DEVICES']==''


@pytest.mark.parametrize('patch',[{'devices':[]},{'devices':[4,4]},{'devices':[True]},
    {'service_batch_queries':True},{'service_batch_queries':16},{'waves_per_macro':0},{'unknown':1}])
def test_invalid_topology(patch):
    with pytest.raises(ValueError):topology.topology(dict(devices=[4,5,6,7],service_batch_queries=32,waves_per_macro=1,**{})|patch)


def fixture(tmp_path,monkeypatch,n=4):
    c,parent=setup(tmp_path,monkeypatch)
    c['selection_inputs']['cache_root']='/data/caches/selection'
    c['manifest_sha256']=package.parent_api.prepare(c,{})['manifest_sha256']
    original=Path(c['output'])
    for name in ('a','b'):
        p=original/f'service-{name}.json';v=json.loads(p.read_text())
        v['policy'].update(max_model_len=4096,tensor_parallel_size=1)
        write(p,v)
    # Parent provenance is a fixture; new package bytes and independent rebuild
    # use the real implementation, including actual generated runner/client.
    m=json.loads((original/'PREPARED.json').read_text())
    m['files']={str(p.relative_to(original)):package.file_hash(p) for p in original.rglob('*') if p.is_file() and p.name!='PREPARED.json'}
    write(original/'PREPARED.json',m)
    c['manifest_sha256']=package.file_hash(original/'PREPARED.json')
    monkeypatch.setattr(package.parent_api,'audit',lambda *a:{'fixture':True})
    monkeypatch.setattr(package,'check_paths',lambda _:None)
    return dict(parent=c,output=str(tmp_path/'multi'),remote_root='/data/multi',
        topology=dict(devices=[7,4,6,5][:n],service_batch_queries=32,waves_per_macro=1))


@pytest.mark.parametrize('n',[1,2,4])
def test_package_reconstructs_and_preserves_source(tmp_path,monkeypatch,n):
    c=fixture(tmp_path,monkeypatch,n);c['manifest_sha256']=package.prepare(c,{})['manifest_sha256']
    assert package.audit(c,{})['replicas']==n
    root=Path(c['output']);m=json.loads((root/'PREPARED.json').read_text());conf=json.loads((root/'config.json').read_text())
    assert m['training_devices']==[] and len(m['inference_services'])==n
    assert (root/'RUNTIME-SOURCES.json').read_bytes()==(Path(c['parent']['output'])/'RUNTIME-SOURCES.json').read_bytes()
    assert conf['macro_client']['max_queries']==n*32
    argv=topology.command(dict(shared_root='/shared',driver_root='/driver',tag='multi-test'),m,conf)
    actual=[v for v in argv if v.startswith('/dev/davinci') and 'manager' not in v]
    assert actual==[f'/dev/davinci{d}:/dev/davinci{d}' for d in sorted(c['topology']['devices'])]
    from tools.reports.context_cached_storage_operations import audit_mounts
    assert audit_mounts(argv,conf,producer=True)['readonly_mounts']>0
    # Complete rehashing cannot hide a changed topology or untested wrapper.
    (root/'selection-client.py').write_text('pass\n')
    m['files']['selection-client.py']=package.file_hash(root/'selection-client.py')
    write(root/'PREPARED.json',m);c['manifest_sha256']=package.file_hash(root/'PREPARED.json')
    with pytest.raises(ValueError,match='reconstruction'):package.audit(c,{})


@pytest.mark.parametrize('n',[1,2,4])
@pytest.mark.parametrize('size',[1,65,128])
def test_parallel_pool_order_tail_and_exact_equivalence(tmp_path,monkeypatch,n,size):
    from test_context_macro_rollout import Client,rows
    from test_context_replica_macro_rollout import pool
    batch=rows(size)
    one=pool(tmp_path/'one',monkeypatch,[Client()],rollout_queries=32)
    multi=pool(tmp_path/'multi',monkeypatch,[Client() for _ in range(n)],rollout_queries=32)
    assert one.score_many(batch,audio_token_id=9)==multi.score_many(batch,audio_token_id=9)
    assert multi.last_receipt['completed_indices']==list(range(size))


@pytest.mark.parametrize('fault',['short','order','policy','timeout'])
def test_four_services_do_not_publish_partial_success(tmp_path,monkeypatch,fault):
    from test_context_macro_rollout import Client,rows
    from test_context_replica_macro_rollout import pool
    clients=[Client() for _ in range(4)];clients[-1].bad=fault
    p=pool(tmp_path,monkeypatch,clients,rollout_queries=32)
    with pytest.raises((ValueError,TimeoutError)):p.score_many(rows(128),audio_token_id=9)
    assert not list(tmp_path.glob('*/COMPLETE.json'))


@pytest.mark.parametrize('n',[1,2,4])
@pytest.mark.parametrize('fault',[None,'child','service','journal','parity'])
def test_configured_output_audit(tmp_path,monkeypatch,n,fault):
    from test_audit_context_train_parity import outputs
    from tools.reports import context_replica_selection_operations as old
    from tools.reports import context_train_macro_parity as producer
    monkeypatch.setattr(producer,'audit_macros',lambda *a,**kw:topology.audit_macros(*a,replicas=n,**kw))
    p,col=outputs(tmp_path,monkeypatch,actual_replicas=n)
    selected=old.read(col/'hf/panel.json')
    monkeypatch.setattr(old,'independent_inputs',lambda *a:(selected['assets'],selected['candidate_ids'],selected['input_receipts'],selected))
    (col/'parity-journal').rename(col/'selection-journal')
    cfg=dict(devices=list(range(4,4+n)),service_batch_queries=32,waves_per_macro=4//n)
    t=topology.topology(cfg)
    c=old.read(p/'parity-macro.json');c.update(policy_id=c['source_policy_id'],contract=c['source_contract'],release='fixture',macro_queries=128)
    c['macro_client']['replicas']=[{} for _ in range(n)];write(p/'parity-macro.json',c)
    m=old.read(p/'PREPARED.json');m.update(source_step=318,training_devices=[],evaluation_topology=cfg,inference_services=t['inference_services'])
    seed=old.read(p/'service-a.json');ready=old.read(col/'service-a/ready.json')
    m['layout']['services']={s['name']:f"service-{s['name']}.json" for s in t['inference_services']};write(p/'PREPARED.json',m)
    for s in t['inference_services']:
        write(p/m['layout']['services'][s['name']],seed)
        write(col/f"service-{s['name']}/ready.json",dict(ready,physical_devices=[s['device']]))
        write(col/f"service-{s['name']}/terminal.json",{'exit_code':0})
    superv=dict(passed=True,actor_started=True,actor_exit_code=0,forced_cleanup=False,
        child_exit_codes=[0]*(n+1),services=[dict(name=s['name'],exit_code=0) for s in t['inference_services']])
    if fault=='child':superv['child_exit_codes'][-1]=1
    write(col/'supervision/terminal.json',superv)
    inference=col/'evaluation/inference';records=old.read(inference/'predictions.json')['records']
    parity=old.client.parity(old.read(col/'hf/reference.json'),records,selected,c['policy_id']);write(col/'evaluation/parity.json',parity)
    result=old.read(inference/'result.json')
    write(col/'evaluation/result.json',dict(fixture_score=1,input_receipts_sha256=result['input_receipts_sha256'],
        inference_result_sha256=old.file_hash(inference/'result.json'),parity_sha256=old.file_hash(col/'evaluation/parity.json')))
    monkeypatch.setattr(old.client,'score',lambda *a:{'fixture_score':1})
    monkeypatch.setattr(old,'compare_metrics',lambda a,b: {'exact':a['fixture_score']==b['fixture_score']})
    if fault=='service':write(col/'service-eval0/terminal.json',{'exit_code':1})
    if fault=='parity':write(col/'evaluation/parity.json',{})
    if fault=='journal':
        q=col/'selection-journal/test/evaluation-00000000/INPUT.json';v=old.read(q);v['jobs'][0]['indices']=[999];write(q,v)
    if fault:
        with pytest.raises((ValueError,IndexError)):operations.audit_outputs(p,col)
    else:
        proof=operations.audit_outputs(p,col)
        assert proof['evaluation_replicas']==n and all(proof['per_service_hf_queries'])
