import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tools.reports import prepare_context_reference_job as module


def training(root='/data/bound'):
    return dict(physical_devices=[4,5],reference_source=dict(endpoint=318,population=40513,max_tokens=4096),source_root=root+'/source',source_policy=root+'/initial/policy',source_policy_id='policy',
        source_contract={'token':{'id':99}},base_id='base',release='/shared/release',release_sha256='release',
        audio_token_id=98,encoded_audio_cache='/data/cache/input',audio_mirror_root='/data/mirror/run',
        dual_runtime_identity={'processor_sha256':'processor'},dual_runtime_manifest=root+'/RUNTIME-SOURCES.json',
        macro_client=dict(max_queries=128,rollout_queries=32,runtime_identity={'processor_sha256':'processor'},
            attempt_id='attempt',journal_root=root+'/run/pool-journal',
            replicas=[dict(socket=root+'/service-'+n+'/socket',exchange=root+'/service-'+n+'/exchange') for n in ('a','b')]),
        pins={'/shared/model':'model-hash'})


def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))


def setup(tmp_path,monkeypatch):
    parent=tmp_path/'parent';parent.mkdir()
    c=training();write(parent/'config.json',c)
    script=parent/'source/example.py';script.parent.mkdir();script.write_text('pass\n')
    write(parent/'RUNTIME-SOURCES.json',{'example.py':module.file_hash(script)})
    design=json.loads((Path.cwd()/'configs/experiments/context-two-stage-v0.0.11.json').read_bytes())
    write(parent/'initial/endpoint-commit.json',dict(updates=318,joint_policy={'policy_id':'policy'},contract={'batch_config':design['stage1']['batch_config']}))
    (parent/'run.py').write_text('raise SystemExit("pending")')
    for n in ('a','b'):write(parent/f'service-{n}.json',dict(source_policy=c['source_policy'],pins={},
        socket_path='/data/bound/service-'+n+'/socket',exchange='/data/bound/service-'+n+'/exchange',
        policy={'policy_id':'policy'}))
    m=dict(remote_root='/data/bound',training_devices=[4,5],inference_services=[dict(name='a',device=6),dict(name='b',device=7)],
        layout=dict(configs=['config.json','run.py','RUNTIME-SOURCES.json','PREPARED.json'],
            services={'a':'service-a.json','b':'service-b.json'},directories=[]))
    write(parent/'PREPARED.json',m)
    evidence={}
    for n in ('input_proof','input_receipts'):
        p=tmp_path/(n+'.json');write(p,{'test':n});evidence.update({n:str(p),n+'_sha256':module.file_hash(p)})
    monkeypatch.setattr(module,'audit_parent',lambda *a:dict(runtime_identity=c['dual_runtime_identity']))
    monkeypatch.setattr(module,'load',lambda *a,**k:([{'id':i} for i in range(40513)],{},{}))
    monkeypatch.setattr(module,'input_evidence',lambda *a,**k:({},dict(test='independent-inputs-fixture')))
    monkeypatch.setattr(module,'check_paths',lambda _:None)
    from multimodal_jev import local_checkpoint
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda _:None)
    return dict(parent=dict(output=str(parent),manifest_sha256=module.file_hash(parent/'PREPARED.json')),
        output=str(tmp_path/'job'),remote_root='/data/reference-job',input_evidence=evidence)


def test_build_config_uses_same_runtime_correct_mirror_and_separate_journal():
    t=training();before=copy.deepcopy(t)
    evidence=dict(input_proof_sha256='proof',input_receipts_sha256='stream')
    c=module.build_config(t,'/data/bound',evidence)
    assert t==before
    assert c['audio_mirror_root']=='/data/mirror/run/audio'
    assert c['stage1_endpoint']==318 and c['expected_population']==40513
    assert c['macro_client']['attempt_id']=='reference-attempt'
    assert c['macro_client']['journal_root']=='/data/bound/reference-journal'
    assert c['dual_runtime_identity']==t['dual_runtime_identity']
    assert c['macro_client']['replicas']==t['macro_client']['replicas']


def test_real_package_layout_pins_and_parent_are_preserved(tmp_path,monkeypatch):
    c=setup(tmp_path,monkeypatch);parent=Path(c['parent']['output'])
    before={str(p.relative_to(parent)):module.file_hash(p) for p in parent.rglob('*') if p.is_file()}
    result=module.prepare(c,{})
    assert before=={str(p.relative_to(parent)):module.file_hash(p) for p in parent.rglob('*') if p.is_file()}
    root=Path(c['output']);receipt=json.loads((root/'PREPARED.json').read_text())
    assert result['reference_dispatch_allowed'] is False and receipt['actor_updates']==0
    from multimodal_jev.local_config import load_config
    build,_=load_config(root/'reference-template.json',root/'reference-local.json')
    assert build['output']=='/data/reference-job/reference-build'
    assert build['pins'][build['stage1_commit']]==module.file_hash(root/'initial/endpoint-commit.json')
    for key,digest in build['pins'].items():
        if key.startswith('/data/reference-job/'):
            assert module.file_hash(root/Path(key).relative_to('/data/reference-job'))==digest
    assert (root/'RUNTIME-SOURCES.json').read_bytes()==(parent/'RUNTIME-SOURCES.json').read_bytes()
    assert receipt['files']=={str(p.relative_to(root)):module.file_hash(p) for p in root.rglob('*')
        if p.is_file() and p!=root/'PREPARED.json'}
    assert build==json.loads((root/'reference-build-config.json').read_text())
    proof=module.audit(dict(c,manifest_sha256=result['manifest_sha256']),{})
    assert proof['recordings']==40513 and proof['actor_updates']==0
    assert not proof['reference_dispatch_allowed'] and not proof['npu_executed']


def test_eight_card_reference_binds_all_scoring_devices(tmp_path,monkeypatch):
    c=setup(tmp_path,monkeypatch)
    c['scoring_devices']=list(range(8))
    p=module.prepare(c,{})
    root=Path(c['output'])
    manifest=json.loads((root/'PREPARED.json').read_text())
    client=json.loads((root/'reference-build-config.json').read_text())['macro_client']
    assert not manifest['training_devices']
    assert [x['device'] for x in manifest['inference_services']]==list(range(8))
    assert client['max_queries']==256 and client['rollout_queries']==32
    assert len({r['socket'] for r in client['replicas']})==8
    from tools.reports.context_cached_storage_operations import reference_command
    argv=reference_command(dict(tag='reference-job-test',shared_root='/shared',
        driver_root='/usr/local/Ascend/driver'),manifest,json.loads((root/'config.json').read_text()))
    assert all(f'/dev/davinci{i}:/dev/davinci{i}' in argv for i in range(8))
    audit=dict(c,manifest_sha256=p['manifest_sha256'])
    module.audit(audit,{})
    with pytest.raises(ValueError,match='service scope'):
        module.audit(dict(audit,scoring_devices=list(reversed(range(8)))),{})


def test_generated_runner_starts_two_services_and_only_cpu_builder(tmp_path,monkeypatch):
    c=setup(tmp_path,monkeypatch);module.prepare(c,{})
    root=Path(c['output']);seen=[]
    monkeypatch.setitem(sys.modules,'context_process_supervisor',SimpleNamespace(
        supervise_services=lambda **kw:(seen.append(kw) or {'passed':True})))
    import multimodal_jev.local_runtime_storage as storage
    monkeypatch.setattr(storage,'require_training_storage',lambda _:None)
    # Avoid container-only /jev-tmp while executing actual generated runner.
    source=(root/'run.py').read_text().replace("Path('/jev-tmp')",'Path('+repr(str(tmp_path))+')')
    with pytest.raises(SystemExit) as exc:exec(compile(source,'<reference-runner>','exec'),{'__file__':str(root/'run.py')})
    assert exc.value.code==0 and len(seen)==1
    job=seen[0]
    assert len(job['services'])==2
    assert [s['env']['ASCEND_RT_VISIBLE_DEVICES'] for s in job['services']]==['2','3']
    assert job['actor_env']['ASCEND_RT_VISIBLE_DEVICES']==''
    assert job['actor_command'][1].endswith('build_context_reference_dataset.py')
    assert 'torchrun' not in str(job['actor_command'])


def test_missing_or_changed_input_evidence_never_publishes_manifest(tmp_path,monkeypatch):
    c=setup(tmp_path,monkeypatch)
    Path(c['input_evidence']['input_receipts']).write_text('changed')
    with pytest.raises(ValueError,match='copied independent'):module.prepare(c,{})
    assert not (Path(c['output'])/'PREPARED.json').exists()


@pytest.mark.parametrize('fault',['batch','runner','input','code','layout','endpoint','plain-config','extra'])
def test_self_rehashed_bad_job_is_not_independently_accepted(tmp_path,monkeypatch,fault):
    c=setup(tmp_path,monkeypatch);module.prepare(c,{})
    root=Path(c['output']);manifest=root/'PREPARED.json';receipt=json.loads(manifest.read_text())
    if fault=='batch':
        p=root/'service-a.json';value=json.loads(p.read_text());value['policy']['max_batch_queries']=4;write(p,value)
    elif fault=='runner':(root/'run.py').write_text('raise SystemExit(0)\n')
    elif fault=='input':(root/'reference-inputs/receipts.jsonl').write_text('changed')
    elif fault=='code':(root/'source/example.py').write_text('changed')
    elif fault=='layout':receipt['layout']['directories'].remove('reference-journal')
    elif fault=='endpoint':write(root/'initial/endpoint-commit.json',{'updates':288})
    elif fault=='plain-config':
        p=root/'reference-build-config.json';value=json.loads(p.read_text());value['expected_population']=32;write(p,value)
    else:(root/'unrequested.py').write_text('pass')
    receipt['files']={str(p.relative_to(root)):module.file_hash(p) for p in root.rglob('*') if p.is_file() and p!=manifest}
    write(manifest,receipt)
    with pytest.raises(ValueError):module.audit(dict(c,manifest_sha256=module.file_hash(manifest)),{})


@pytest.mark.parametrize('fault',[None,'config','attempt','remote'])
def test_collected_audit_configuration_preserves_actual_producer_binding(tmp_path,monkeypatch,fault):
    import shutil
    c=setup(tmp_path,monkeypatch);module.prepare(c,{})
    root=Path(c['output']);collection=tmp_path/'collected';collection.mkdir()
    shutil.copyfile(root/'reference-build-config.json',collection/'reference-build-config.json')
    c.update(collection=str(collection),results_audit_output=str(tmp_path/'outputs-audit.json'))
    if fault=='config':write(collection/'reference-build-config.json',{})
    elif fault in ('attempt','remote'):
        value=json.loads((root/'reference-build-config.json').read_text())
        if fault=='attempt':value['macro_client']['attempt_id']='../../escape'
        else:value['stage1_commit']='/other/commit.json'
        write(root/'reference-build-config.json',value);write(collection/'reference-build-config.json',value)
    if fault:
        with pytest.raises(ValueError):module.collected_audit_config(c)
    else:
        actual=module.collected_audit_config(c)
        assert actual['journal']==str(collection/'reference-journal/reference-attempt')
        assert actual['stage1_commit']==str(root/'initial/endpoint-commit.json')
        assert actual['input_receipts']==str(root/'reference-inputs/receipts.jsonl')
        assert actual['build_config_sha256']==module.file_hash(root/'reference-build-config.json')


@pytest.mark.parametrize('fault',[None,'transport','collection','supervisor','argv','manifest','readonly','declaration'])
def test_results_audit_requires_exact_terminal_and_transport_before_replay(tmp_path,monkeypatch,fault):
    import shutil
    from tools.reports.prepare_context_replica_training_candidate import IMAGE
    c=setup(tmp_path,monkeypatch);prepared=module.prepare(c,{})
    root=Path(c['output']);m=json.loads((root/'PREPARED.json').read_text())
    collection=tmp_path/'collected';collection.mkdir()
    for name in m['layout']['configs']+list(m['layout']['services'].values()):
        shutil.copyfile(root/name,collection/name)
    write(collection/'supervision/terminal.json',dict(passed=True,actor_started=True,actor_exit_code=0,forced_cleanup=False,
        services=[dict(name=n,exit_code=0) for n in ('a','b')]))
    from tools.reports.context_cached_storage_operations import reference_command,mounts
    c['deployment']=dict(tag='reference-job-test',shared_root='/shared',driver_root='/usr/local/Ascend/driver')
    argv=reference_command(c['deployment'],m,json.loads((root/'config.json').read_text()))
    def values(flag):return [argv[i+1] for i,v in enumerate(argv) if v==flag]
    state=dict(Status='exited',ExitCode=0,OOMKilled=False)
    handle=dict(argv=argv,container_id='a'*64,manifest_sha256=prepared['manifest_sha256'],
        layout=m['layout'],remote_root=c['remote_root'],image=IMAGE)
    observed=dict(Id=handle['container_id'],Image=IMAGE,State=state,
        HostConfig=dict(Privileged=False,NetworkMode='none',Binds=values('-v'),
            Devices=[dict(PathOnHost=v.split(':')[0],PathInContainer=v.split(':')[1],CgroupPermissions='rwm')
                     for v in values('--device')]),
        Config=dict(Entrypoint=values('--entrypoint'),Cmd=argv[argv.index(IMAGE)+1:],Env=values('-e')),
        Mounts=[dict(Source=src,Destination=dst,RW=not ro,Type='bind') for src,dst,ro in mounts(argv)])
    transport=dict(container_id=handle['container_id'],image=IMAGE,remote_root=c['remote_root'],state=state,
        status='transport_complete_independent_audit_required',bad_pins=[],bad_package=[],launch_binding_mismatches=[],
        files={str(p.relative_to(collection)):module.file_hash(p) for p in collection.rglob('*') if p.is_file()})
    if fault=='transport':transport['bad_pins']=['changed']
    elif fault=='collection':(collection/'extra').write_text('extra')
    elif fault=='supervisor':
        p=collection/'supervision/terminal.json';v=json.loads(p.read_text());v['forced_cleanup']=True;write(p,v)
        transport['files']['supervision/terminal.json']=module.file_hash(p)
    elif fault=='argv':handle['argv']=argv+['other']
    elif fault=='manifest':handle['manifest_sha256']='0'*64
    elif fault=='readonly':
        next(v for v in observed['Mounts'] if v['Destination']=='/data/cache/input')['RW']=True
    elif fault=='declaration':c['deployment']['shared_root']='/other-shared'
    for name,value in [('launch',handle),('inspect',observed),('transport',transport)]:
        p=tmp_path/(name+'.json');write(p,value);c[name]=str(p)
    c.update(collection=str(collection),manifest_sha256=prepared['manifest_sha256'],expected_argv=argv,
        results_audit_output=str(tmp_path/'output-audit.json'))
    calls=[]
    monkeypatch.setattr(module,'replay_reference_outputs',lambda c,p:(calls.append(c) or {'test':'replay-stub'}))
    if fault:
        with pytest.raises(ValueError):module.audit_results(c,{})
        assert not calls
    else:
        result=module.audit_results(c,{})
        assert len(calls)==1 and not result['numerical_parity_verified']
        assert result['storage_audit']['actual_readonly_mounts_verified']
        assert result['output_audit']=={'test':'replay-stub'}
