import copy
import json
from pathlib import Path

import pytest

from tools.reports import context_reference_job_operations as module


def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))


def fixture(tmp_path,monkeypatch):
    from tools.reports import context_release_deployment
    monkeypatch.setattr(context_release_deployment,'audit',lambda *a:{'fixture':'external release mount admission'})
    from test_prepare_context_reference_job import setup
    from tools.reports import prepare_context_reference_job as reference
    from tools.reports import prepare_context_train_parity_job as parity
    spec=setup(tmp_path,monkeypatch);parent=Path(spec['parent']['output'])
    training=module.read(parent/'config.json');training.update(model='/shared/model',initial_adapter='/data/bound/initial/adapter')
    write(parent/'config.json',training)
    for name in ('context_train_macro_parity.py','context_train_parity_inputs.py'):
        p=parent/'source/tools/reports'/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(Path(parity.__file__).with_name(name).read_bytes())
    write(parent/'RUNTIME-SOURCES.json',{str(p.relative_to(parent/'source')):module.file_hash(p)
        for p in (parent/'source').rglob('*.py')})
    write(parent/'initial/policy/MANIFEST.json',{'policy':'fixture'})
    m=reference.json.loads((parent/'PREPARED.json').read_text());m['image']=module.lifecycle.shared.IMAGE
    write(parent/'PREPARED.json',m)
    spec['parent']['manifest_sha256']=module.file_hash(parent/'PREPARED.json')
    spec['manifest_sha256']=reference.prepare(spec,{})['manifest_sha256']
    inputs=tmp_path/'inputs'
    write(inputs/'inputs.json',{'source':{'release_sha256':training['release_sha256'],
        'processor_sha256':training['dual_runtime_identity']['processor_sha256']}})
    write(inputs/'MANIFEST.json',{'fixture':True})
    input_spec=dict(output=str(inputs),manifest_sha256=module.file_hash(inputs/'MANIFEST.json'),
        cache_root=training['encoded_audio_cache'],**spec['input_evidence'])
    monkeypatch.setattr(parity,'audit_parent',lambda *a:{'fixture':'NPU-parent'})
    monkeypatch.setattr(parity,'audit_inputs',lambda *a:{'fixture':'input-audit'})
    monkeypatch.setattr(parity,'check_paths',lambda _:None)
    ps=dict(parent=spec['parent'],inputs=input_spec,output=str(tmp_path/'parity'),remote_root='/data/parity-job')
    ps['manifest_sha256']=parity.prepare(ps,{})['manifest_sha256']
    proof=dict(runtime_provenance_verified=True,comparison=dict(exact=True,queries=256,decisions=60),
        actor_updates=0,container_id='a'*64,transport_sha256='b'*64)
    calls=[]
    monkeypatch.setattr(module,'audit_parity',lambda *a:(calls.append('independent_parity_replay') or copy.deepcopy(proof)))
    c=dict(reference_job=spec,parity_audit={'parity_package':ps})
    return c,proof,calls


def test_actual_same_parent_packages_bind_to_replayed_numerical_proof(tmp_path,monkeypatch):
    c,_,calls=fixture(tmp_path,monkeypatch)
    result=module.admission(c)
    assert calls==['independent_parity_replay']
    assert result['source_policy_id']=='policy' and result['finite_panel_only']
    assert not result['training_dispatch_allowed'] and not result['full_population_parity']


@pytest.mark.parametrize('fault',['parent','not-exact','runtime-unverified','population','width','updates',
    'policy','runtime','release','proof','receipts','commit','source','policy-asset'])
def test_other_or_self_reported_parity_never_admits_reference(tmp_path,monkeypatch,fault):
    c,proof,_=fixture(tmp_path,monkeypatch)
    ps=c['parity_audit']['parity_package'];root=Path(ps['output'])
    if fault=='parent':ps['parent']=dict(ps['parent'],manifest_sha256='other')
    elif fault=='not-exact':proof['comparison']['exact']=False
    elif fault=='runtime-unverified':proof['runtime_provenance_verified']=False
    elif fault=='population':proof['comparison']['queries']=32
    elif fault=='width':proof['comparison']['decisions']=14
    elif fault=='updates':proof['actor_updates']=2
    elif fault in ('policy','runtime','release'):
        p=root/'parity-macro.json';v=module.read(p)
        v[{'policy':'source_policy_id','runtime':'dual_runtime_identity','release':'release_sha256'}[fault]]='other'
        write(p,v)
    elif fault in ('proof','receipts'):
        ps['inputs']['input_proof_sha256' if fault=='proof' else 'input_receipts_sha256']='other'
    else:
        p=root/{'commit':'initial/endpoint-commit.json','source':'RUNTIME-SOURCES.json',
            'policy-asset':'initial/policy/MANIFEST.json'}[fault]
        write(p,{'other':'asset'})
    with pytest.raises(ValueError):module.admission(c)


def test_failed_independent_parity_is_propagated_before_launch(tmp_path,monkeypatch):
    c,_,_=fixture(tmp_path,monkeypatch)
    def failure(*a):raise ValueError('actual logits differ')
    monkeypatch.setattr(module,'audit_parity',failure)
    with pytest.raises(ValueError,match='actual logits'):module.validate(c)


def test_reference_dispatch_does_not_accept_finite_inference_flag():
    with pytest.raises(ValueError,match='full TRAIN reference'):
        module.launch({'finite_inference_dispatch_allowed':True},{})


def test_reference_validation_reuses_shared_scope_after_gate(tmp_path,monkeypatch):
    from tools.reports import context_release_deployment
    monkeypatch.setattr(context_release_deployment,'audit',lambda *a:{'fixture':'external release mount admission'})
    from test_context_train_parity_operations import fixture as lifecycle_fixture
    c=lifecycle_fixture(tmp_path,monkeypatch)
    c['reference_job']=c.pop('parity_package');seen=[]
    monkeypatch.setattr(module,'admission',lambda c:seen.append(c['reference_job']))
    m,config,_=module.validate(c)
    assert len(seen)==1 and m['remote_root']==c['reference_job']['remote_root']
    c['host']='unsafe host'
    with pytest.raises(ValueError):module.validate(c)


def test_reference_launch_uses_its_own_gate_in_shared_lifecycle(tmp_path,monkeypatch):
    from test_context_train_parity_operations import fixture as lifecycle_fixture
    c=lifecycle_fixture(tmp_path,monkeypatch)
    c['reference_job']=c.pop('parity_package');c['reference_dispatch_allowed']=True
    called=[]
    def fail(c):called.append(c);raise ValueError('reference gate refused')
    monkeypatch.setattr(module,'validate',fail)
    with pytest.raises(ValueError,match='reference gate refused'):module.launch(c,{})
    assert len(called)==1 and called[0]['finite_inference_dispatch_allowed'] is True


@pytest.mark.parametrize('fault',[None,'wait','archive','config','service','parity'])
def test_reference_result_gate_binds_wait_archive_and_independent_output_audit(tmp_path,monkeypatch,fault):
    c,_,_=fixture(tmp_path,monkeypatch);spec=c['reference_job'];root=Path(spec['output'])
    m=module.read(root/'PREPARED.json');cid='d'*64
    handle=dict(container_id=cid,config_sha256=module.file_hash(root/'config.json'),
        services={n:module.file_hash(root/p) for n,p in m['layout']['services'].items()})
    archive=tmp_path/'transport.tar';archive.write_bytes(b'fixture transport')
    transport=dict(archive_sha256=module.file_hash(archive));wait=dict(container_id=cid,exit_code=0)
    if fault=='wait':wait['container_id']='wrong'
    elif fault=='archive':archive.write_bytes(b'changed')
    elif fault=='config':handle['config_sha256']='wrong'
    elif fault=='service':handle['services']['a']='wrong'
    elif fault=='parity':
        def fail(*a):raise ValueError('parity evidence corrupted')
        monkeypatch.setattr(module,'audit_parity',fail)
    for name,value in [('launch',handle),('transport',transport),('wait',wait)]:
        p=tmp_path/(name+'.json');write(p,value);c[name]=str(p)
    c.update(archive=str(archive),deployment=dict(tag='reference-test',shared_root='/shared',
        driver_root='/usr/local/Ascend/driver'),inspect='fixture-inspect',collection='fixture-collected',
        results_audit_output=str(tmp_path/'independent-reference.json'))
    calls=[]
    # Exact transport and full output replay are separately exercised through
    # prepare_context_reference_job tests; check this gate composes them intact.
    monkeypatch.setattr(module,'audit_results',lambda spec,p:(calls.append(spec) or {'output_audit':'fixture'}))
    if fault:
        with pytest.raises(ValueError):module.audit(c,{})
        assert not calls
    else:
        result=module.audit(c,{})
        assert len(calls)==1 and calls[0]['results_audit_output']==c['results_audit_output']
        assert calls[0]['expected_argv'][-1]==spec['remote_root']+'/run.py'
        assert not result['training_dispatch_allowed'] and not result['full_population_parity']
        assert result['numerical_admission']['parity_container_id']=='a'*64
