from pathlib import Path

import pytest

from tools.reports import prepare_objective_recovery as module
from tests.test_context_cached_recovery import fixture as cached_fixture,write
from tests.test_objective_plugin import ROOT,binding


def fixture(tmp_path,monkeypatch):
    c,run=cached_fixture(tmp_path,monkeypatch)
    parent=Path(c['parent_audit']['training_package']['output'])
    cfg=module.cached.read(parent/'config.json')
    extension=parent/'objective-extension/extensions/stage2_objective.py'
    extension.parent.mkdir(parents=True);extension.write_bytes((ROOT/'extensions/stage2_objective.py').read_bytes())
    cfg['objective_plugin']=dict(binding(extension),path='/data/train/objective-extension/extensions/stage2_objective.py')
    cfg['objective_extension']=dict(root='/data/train/objective-extension',files={'extensions/stage2_objective.py':module.cached.file_hash(extension)})
    cfg['objective_trace']=True;write(parent/'config.json',cfg)
    commit=module.cached.read(run/'checkpoint-0001-committed.json')
    m=module.cached.read(parent/'PREPARED.json')
    commit['contract']=module.expected_contract(cfg,module.cached.mounted_names(m))
    write(run/'checkpoint-0001-committed.json',commit)
    transport=Path(c['parent_audit']['transport']);value=module.cached.read(transport)
    value['files']['run/checkpoint-0001-committed.json']=module.cached.file_hash(run/'checkpoint-0001-committed.json')
    write(transport,value)
    def parent(*a):
        return dict(status='finite_objective_training_outputs_audited',world_size=2,
            manifest_sha256=c['parent_audit']['training_package']['manifest_sha256'],container_id='a'*64,
            objective_trace_audit={'status':'finite_objective_traces_replayed'})
    monkeypatch.setattr(module,'audit_parent',parent)
    return c,run


def test_recovery_preserves_plugin_and_reference(tmp_path,monkeypatch):
    c,run=fixture(tmp_path,monkeypatch)
    original_functions=(module.cached.prepare,module.cached.inputs,module.cached.expected_contract)
    c['manifest_sha256']=module.prepare(c,{})['manifest_sha256']
    result=module.audit(c,{})
    assert result['objective_identity_preserved'] and not result['npu_executed']
    root=Path(c['output']);new=module.cached.read(root/'config.json')
    old=module.cached.read(Path(c['parent_audit']['training_package']['output'])/'config.json')
    assert new['reference_dataset']==old['reference_dataset']
    assert new['objective_plugin']==dict(old['objective_plugin'],path='/data/recovery/objective-extension/extensions/stage2_objective.py')
    assert 'objective-extension/bootstrap.py' in (root/'run.py').read_text()
    assert new['resume_root']=='/data/recovery/resume' and new['start_update']==1
    assert (module.cached.prepare,module.cached.inputs,module.cached.expected_contract)==original_functions


@pytest.mark.parametrize('fault',['options','hash','file','reference','extra_base'])
def test_rehashed_recovery_cannot_change_objective_or_reference(tmp_path,monkeypatch,fault):
    c,_=fixture(tmp_path,monkeypatch);module.prepare(c,{})
    root=Path(c['output']);cfg=module.cached.read(root/'config.json')
    if fault=='options':cfg['objective_plugin']['options']={'new_weight':2}
    elif fault=='hash':cfg['objective_plugin']['sha256']='f'*64
    elif fault=='reference':cfg['reference_dataset']['root']='/other/reference'
    elif fault=='file':(root/'objective-extension/extensions/stage2_objective.py').write_text('pass')
    else:(root/'resume/base.bin').write_bytes(b'base')
    write(root/'config.json',cfg)
    m=module.cached.read(root/'PREPARED.json')
    m['files']={str(p.relative_to(root)):module.cached.file_hash(p) for p in root.rglob('*') if p.is_file() and p!=root/'PREPARED.json'}
    write(root/'PREPARED.json',m);c['manifest_sha256']=module.cached.file_hash(root/'PREPARED.json')
    with pytest.raises(ValueError):module.audit(c,{})


def test_missing_parent_trace_proof_refuses_prepare(tmp_path,monkeypatch):
    c,_=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(module,'audit_parent',lambda *a:dict(status='finite_objective_training_outputs_audited'))
    with pytest.raises(ValueError,match='trace'):module.prepare(c,{})
    assert not Path(c['output']).exists()


@pytest.mark.parametrize('fault',[None,'objective','adam','reference'])
def test_recovery_comparison_keeps_optimizer_and_actual_objective(tmp_path,fault):
    import torch
    from tests.test_context_cached_recovery import replay_fixture
    from tools.reports.audit_objective_recovery import compare
    left,right=replay_fixture(tmp_path)
    for root in (left,right):
        for rank in range(2):
            write(root/f'rank-{rank}/objective-traces/objective-0001.json',
                  dict(objective={'sha256':'same'},rewards=[1,.5],loss=.2))
    if fault=='objective':write(right/'rank-1/objective-traces/objective-0001.json',{'loss':.3})
    elif fault=='adam':
        p=right/'rank-1/checkpoint-0002.pt';v=torch.load(p,weights_only=True)
        v['optimizer']['state'][0]['exp_avg'][0]+=.1;torch.save(v,p)
    elif fault=='reference':
        p=right/'rollout-0001.json';v=module.cached.read(p);v['fixed_reference_policy_id']='wrong';write(p,v)
    if fault:
        with pytest.raises(ValueError):compare(left,right)
    else:assert compare(left,right)['exact_objective_traces']


def test_recovery_lifecycle_keeps_readonly_resume(tmp_path,monkeypatch):
    from tools.reports import context_objective_recovery_operations as ops
    from tools.reports.context_objective_storage import mounts
    c,_=fixture(tmp_path,monkeypatch);c['manifest_sha256']=module.prepare(c,{})['manifest_sha256']
    monkeypatch.setattr(ops.shared,'bootstrap_evidence',lambda _:None)
    config=dict(training_package=c,package=c['output'],manifest_sha256=c['manifest_sha256'],
        host='host',tag='objective-replay',shared_root='/shared',driver_root='/driver',
        operations_root=str(tmp_path/'ops'),lease=str(tmp_path/'lease'))
    m,cfg,_=ops.validate(config);argv=ops.shared.command(config,m,cfg)
    assert ('/data/recovery/resume','/data/recovery/resume',True) in mounts(argv)
    assert ('/data/recovery/objective-extension','/data/recovery/objective-extension',True) in mounts(argv)
    with pytest.raises(ValueError,match='explicit'):ops.launch(config,{})


@pytest.mark.parametrize('extra',[False,True])
def test_recovery_output_replays_only_second_update(tmp_path,monkeypatch,surrogate,extra):
    from tests.test_objective_training_audit import setup
    from tools.reports import audit_objective_recovery as audit
    from tools.reports import audit_objective_training as prefix
    p,c,plugin=setup(tmp_path,monkeypatch)
    cfg=module.cached.read(p/'config.json');cfg['start_update']=1;write(p/'config.json',cfg)
    for path in (c/'run').glob('rank-*/objective-traces/objective-0000.json'):path.unlink()
    def checked(package,collection,*,reference_root):
        return dict(status='fixture_original_recovery_audit',world_size=3)
    monkeypatch.setattr(audit.cached,'audit_outputs',checked)
    monkeypatch.setattr(audit,'derive',prefix.derive)
    if extra:
        write(c/'run/rank-0/objective-traces/extra.json',{})
        with pytest.raises(ValueError,match='trace population'):
            audit.audit_outputs(p,c,reference_root='fixture',loss_atol=0.)
    else:
        result=audit.audit_outputs(p,c,reference_root='fixture',loss_atol=0.)
        assert result['objective_trace_audit']['updates'][0]['update']==1
        assert len(result['objective_trace_audit']['updates'])==1


from tests.test_objective_plugin import surrogate
