import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tools.reports import prepare_context_cached_training as module


def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))


def test_tail_package_requires_sealed_entry_and_preserves_parent(tmp_path,monkeypatch):
    from test_context_cached_tail_window import config as tail_config
    c,_,_,_=setup(tmp_path,monkeypatch)
    c['acceptance_kind']='cached_reference_tail_admission'
    parent=Path(c['parent']['output'])
    config=module.read(parent/'config.json');config.update(tail_config())
    write(parent/'config.json',config)
    with pytest.raises(ValueError,match='already belong'):module.prepare(c,{})
    assert not Path(c['output']).exists()
    sources=module.read(parent/'RUNTIME-SOURCES.json')
    project=Path(__file__).resolve().parents[1]
    for name in ('tools/reports/context_cached_tail_trainer.py',
                 'src/multimodal_jev/context_cached_tail_window.py'):
        p=parent/'source'/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes((project/name).read_bytes());sources[name]=module.file_hash(p)
    write(parent/'RUNTIME-SOURCES.json',sources)
    before={str(p):module.file_hash(p) for p in parent.rglob('*') if p.is_file()}
    result=module.prepare(c,{});c['manifest_sha256']=result['manifest_sha256']
    assert module.audit(c,{})['actor_updates']==2
    assert before=={str(p):module.file_hash(p) for p in parent.rglob('*') if p.is_file()}
    assert 'context_cached_tail_trainer.py' in (Path(c['output'])/'run.py').read_text()
    assert module.read(Path(c['output'])/'config.json')['experiment_kind']==c['acceptance_kind']
    c['acceptance_kind']='engineering_prefix'
    with pytest.raises(ValueError):module.audit(c,{})


def setup(tmp_path,monkeypatch):
    from test_context_reference_dataset import fixture
    from multimodal_jev.context_reference_dataset import digest,publish
    from tools.reports.context_cached_reference_trainer import BASELINE_OBJECTIVE
    _,kw,_,_,record=fixture(tmp_path,monkeypatch)
    population={str(i):record['request_id'] for i in range(40513)}
    identity=dict(kw['expected_identity'],population_sha256=digest(population))
    collection=tmp_path/'collection';dataset=collection/'reference-build/dataset'
    publication=publish(dataset,identity=identity,records=[record],population=population,producer={})
    result=dict(publication,identity=identity,population=population)
    write(collection/'reference-build/result.json',result)
    proof=dict(status='reference_dataset_journal_audited',policy_id=identity['policy_id'],
        recordings=40513,manifest_sha256=publication['manifest_sha256'])
    parent=tmp_path/'parent';parent.mkdir()
    (parent/'run.py').write_text('raise SystemExit("not admitted")\n')
    source=parent/'source/test.py';source.parent.mkdir();source.write_text('pass')
    write(parent/'RUNTIME-SOURCES.json',{'test.py':module.file_hash(source)})
    config=dict(source_root='/data/parent/source',source_policy='/data/parent/initial/policy',
        source_policy_id=identity['policy_id'],base_id=identity['base_id'],source_contract=identity['contract'],
        dual_runtime_identity=identity['runtime_identity'],release='/shared/release',release_sha256=identity['release_sha256'],
        stage2_objective=dict(BASELINE_OBJECTIVE),algorithm=BASELINE_OBJECTIVE['algorithm'],dual_arm='joint',
        pins={'/shared/model':'external'})
    write(parent/'config.json',config)
    for name in ('a','b'):write(parent/f'service-{name}.json',dict(socket_path='/data/parent/service-'+name+'/socket',pins={}))
    m=dict(remote_root='/data/parent',training_devices=[4,5],inference_services=[dict(name='a',device=6),dict(name='b',device=7)],
        layout=dict(configs=['config.json','PREPARED.json','run.py','RUNTIME-SOURCES.json'],
            services={n:f'service-{n}.json' for n in ('a','b')},directories=[]))
    write(parent/'PREPARED.json',m)
    job=tmp_path/'job';write(job/'reference-build-config.json',{'output':'/data/producer/reference-build'})
    c=dict(parent=dict(output=str(parent),manifest_sha256='parent'),
        reference_job=dict(output=str(job),collection=str(collection),remote_root='/data/producer',manifest_sha256='job'),
        output=str(tmp_path/'train'),remote_root='/data/train')
    monkeypatch.setattr(module,'audit_parent',lambda *a:{'scope':'fixture'})
    monkeypatch.setattr(module,'audit_results',lambda *a:{'output_audit':proof})
    monkeypatch.setattr(module,'load',lambda *a,**k:([dict(id=key) for key in population],
        {'schema':{'candidates':[{'id':key} for key in identity['candidate_ids']]}},{}))
    monkeypatch.setattr(module,'check_paths',lambda _:None)
    return c,config,result,proof


def test_preparation_uses_real_readonly_results_without_copying_or_modifying_them(tmp_path,monkeypatch):
    c,_,result,_=setup(tmp_path,monkeypatch)
    paths=[Path(c['parent']['output']),Path(c['reference_job']['collection'])]
    before={str(p):module.file_hash(p) for root in paths for p in root.rglob('*') if p.is_file()}
    receipt=module.prepare(c,{})
    assert before=={str(p):module.file_hash(p) for root in paths for p in root.rglob('*') if p.is_file()}
    root=Path(c['output']);config=module.read(root/'config.json')
    assert receipt['training_dispatch_allowed'] is False
    assert config['reference_dataset']['root']=='/data/producer/reference-build/dataset'
    assert config['reference_dataset']['population']==result['population']
    assert config['stop_update']==2 and config['checkpoint_interval']==1
    assert config['source_policy']=='/data/train/initial/policy'
    assert 'reference_cache' not in config
    assert not list(root.rglob('records.json'))
    assert config['pins'][config['reference_dataset']['root']+'/MANIFEST.json']==result['manifest_sha256']
    assert (root/'RUNTIME-SOURCES.json').read_bytes()==(paths[0]/'RUNTIME-SOURCES.json').read_bytes()
    # Execute generated launcher with only the actual process supervisor replaced.
    calls=[]
    monkeypatch.setitem(sys.modules,'context_process_supervisor',SimpleNamespace(
        supervise_services=lambda **kw:(calls.append(kw) or {'passed':True})))
    import multimodal_jev.local_runtime_storage as storage
    monkeypatch.setattr(storage,'require_training_storage',lambda _:None)
    script=(root/'run.py').read_text().replace("Path('/jev-tmp')",'Path('+repr(str(tmp_path))+')')
    with pytest.raises(SystemExit) as exc:exec(compile(script,'<test>','exec'),{'__file__':str(root/'run.py')})
    assert exc.value.code==0
    call=calls[0]
    assert '--nproc-per-node=2' in call['actor_command']
    assert any(p.endswith('context_cached_reference_trainer.py') for p in call['actor_command'])
    assert call['actor_env']['ASCEND_RT_VISIBLE_DEVICES']=='0,1'
    assert [s['env']['ASCEND_RT_VISIBLE_DEVICES'] for s in call['services']]==['2','3']


@pytest.mark.parametrize('fault',['policy','runtime','release','population','order','audit','corrupt','overlap'])
def test_wrong_reference_rejected_before_training_package_publication(tmp_path,monkeypatch,fault):
    c,training,result,proof=setup(tmp_path,monkeypatch)
    collection=Path(c['reference_job']['collection']);path=collection/'reference-build/result.json'
    if fault in ('policy','runtime','release','order'):
        if fault=='policy':result['identity']['policy_id']='d'*64
        elif fault=='runtime':result['identity']['runtime_identity']['profile']='other'
        elif fault=='release':result['identity']['release_sha256']='e'*64
        else:result['identity']['candidate_ids'].reverse()
    elif fault=='population':result['population'].pop('0')
    elif fault=='audit':proof['manifest_sha256']='wrong'
    elif fault=='corrupt':(collection/'reference-build/dataset/records.json').write_text('{}')
    else:c['remote_root']='/data/producer/reference-build/dataset/child'
    write(path,result)
    with pytest.raises(ValueError):module.prepare(c,{})
    assert not Path(c['output']).exists()


def reseal(c):
    """Rehash modified payload and pins: hashes alone must not grant admission."""
    root=Path(c['output']);m=module.read(root/'PREPARED.json')
    names={'config.json',*m['layout']['services'].values()}
    pins=module.read(root/'config.json')['pins']
    pins={k:v for k,v in pins.items() if not Path(k).is_relative_to(c['remote_root'])}
    pins.update({c['remote_root']+'/'+str(p.relative_to(root)):module.file_hash(p)
        for p in root.rglob('*') if p.is_file() and str(p.relative_to(root)) not in names|{'PREPARED.json'}})
    for name in names:
        config=module.read(root/name);config['pins']=pins;write(root/name,config)
    m['files']={str(p.relative_to(root)):module.file_hash(p) for p in root.rglob('*')
        if p.is_file() and p.name!='PREPARED.json'}
    write(root/'PREPARED.json',m);c['manifest_sha256']=module.file_hash(root/'PREPARED.json')


def test_independent_audit_reloads_reference_and_reconstructs_package(tmp_path,monkeypatch):
    c,_,_,_=setup(tmp_path,monkeypatch)
    prepared=module.prepare(c,{'invocation':'prepare'})
    c['manifest_sha256']=prepared['manifest_sha256']
    result=module.audit(c,{'invocation':'audit'})
    assert result['recordings']==40513 and result['actor_updates']==2
    assert result['npu_executed'] is False and result['finite_npu_dispatch_allowed'] is False
    assert result['reference_manifest_sha256']==prepared['reference_manifest_sha256']


def test_cross_host_reference_keeps_producer_identity_and_content_pins(tmp_path,monkeypatch):
    c,_,result,_=setup(tmp_path,monkeypatch)
    c['reference_destination']='/data/reference-copy'
    c['manifest_sha256']=module.prepare(c,{})['manifest_sha256']
    cfg=module.read(Path(c['output'])/'config.json')
    assert cfg['reference_dataset']==dict(root='/data/reference-copy',
        manifest_sha256=result['manifest_sha256'],identity=result['identity'],
        population=result['population'])
    dataset=Path(c['reference_job']['collection'])/'reference-build/dataset'
    for name in ('MANIFEST.json','records.json','population.json'):
        assert cfg['pins']['/data/reference-copy/'+name]==module.file_hash(dataset/name)
        assert '/data/producer/reference-build/dataset/'+name not in cfg['pins']
    module.audit(c,{})
    with pytest.raises(ValueError,match='configuration differs'):
        module.audit(dict(c,reference_destination='/data/unverified-copy'),{})


@pytest.mark.parametrize('fault',['budget','online-reference','service','source','runner','audit',
    'extra','layout','topology','admission','manifest','symlink','population'])
def test_independent_audit_rejects_rehashed_substitutions(tmp_path,monkeypatch,fault):
    c,_,_,_=setup(tmp_path,monkeypatch)
    c['manifest_sha256']=module.prepare(c,{})['manifest_sha256'];root=Path(c['output'])
    if fault in ('budget','online-reference','population'):
        config=module.read(root/'config.json')
        if fault=='budget':config['stop_update']=634
        elif fault=='online-reference':config['reference_cache']='/data/online'
        else:config['reference_dataset']['population'].pop('0')
        write(root/'config.json',config)
    elif fault=='service':
        service=module.read(root/'service-a.json');service['socket_path']='/data/other/socket'
        write(root/'service-a.json',service)
    elif fault=='source':(root/'source/test.py').write_text('raise RuntimeError()')
    elif fault=='runner':
        p=root/'run.py';p.write_text(p.read_text().replace('context_cached_reference_trainer.py','online.py'))
    elif fault=='audit':write(root/'REFERENCE-RESULT-AUDIT.json',{'output_audit':{'status':'forged'}})
    elif fault=='extra':(root/'extra.py').write_text('pass')
    elif fault=='symlink':
        p=root/'source/test.py';p.unlink();p.symlink_to(Path(c['parent']['output'])/'source/test.py')
    elif fault in ('layout','topology','admission'):
        m=module.read(root/'PREPARED.json')
        if fault=='layout':m['layout']['directories']=[]
        elif fault=='topology':m['training_devices']=[6,7]
        else:m['finite_npu_dispatch_allowed']=True
        write(root/'PREPARED.json',m)
    else:c['manifest_sha256']='0'*64
    if fault!='manifest':reseal(c)
    with pytest.raises(ValueError):module.audit(c,{})
