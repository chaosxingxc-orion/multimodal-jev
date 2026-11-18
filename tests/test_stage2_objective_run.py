import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools.reports import prepare_stage2_objective_run as module
from tools.reports import stage2_objective_run_operations as operations
from test_objective_training_candidate import setup,write


def fixture(tmp_path,monkeypatch):
    from tools.reports import prepare_objective_training_candidate as prefix
    c,parent=setup(tmp_path,monkeypatch);c['manifest_sha256']=prefix.prepare(c,{})['manifest_sha256']
    root=Path(c['output']);config=json.loads((root/'config.json').read_text())
    config.update(checkpoint_interval=1,engineering_prefix=True,training_dispatch_allowed=False,stage='task',global_batch=128,microbatch=8,seed=317,algorithm='dual_credit_grpo_floor_clipped_v1',dual_arm='joint',
        batch_config=dict(training_devices=2,tensor_parallel=1,pipeline_parallel=1,per_dp_replica_batch=64,
            microbatch=8,inference_instances=2,inference_batch_queries=32,source_queries=40513,epochs=2,efficiency_steps=100))
    write(root/'config.json',config)
    m=json.loads((root/'PREPARED.json').read_text());m.update(training_devices=[4,5],inference_services=[dict(name='a',device=6),dict(name='b',device=7)])
    m['files']['config.json']=module.prefix.sha((root/'config.json').read_bytes());write(root/'PREPARED.json',m)
    c['manifest_sha256']=module.file_hash(root/'PREPARED.json')
    monkeypatch.setattr(module.prefix,'audit',lambda *a:dict(status='fixture_prefix_audited'))
    monkeypatch.setattr(module,'check_paths',lambda _:None)
    return dict(prefix_package=c,output=str(tmp_path/'run-package'),remote_root='/data/next',kind='formal')


@pytest.mark.parametrize('kind,updates',[('tail',2),('efficiency',100),('formal',634)])
def test_run_package_preserves_plugin_forward_and_reference(tmp_path,monkeypatch,kind,updates):
    c=fixture(tmp_path,monkeypatch);c['kind']=kind
    c['manifest_sha256']=module.prepare(c,{})['manifest_sha256'];module.audit(c,{})
    root=Path(c['output']);old=Path(c['prefix_package']['output']);config=module.read(root/'config.json')
    assert config['stop_update']==updates and config['objective_trace'] is False
    assert config['reference_dataset']==module.read(old/'config.json')['reference_dataset']
    for p in (old/'objective-extension').rglob('*'):
        if p.is_file():assert p.read_bytes()==(root/p.relative_to(old)).read_bytes()
    assert (old/'RUNTIME-SOURCES.json').read_bytes()==(root/'RUNTIME-SOURCES.json').read_bytes()
    assert 'objective-' in (root/'run.py').read_text()
    if kind=='formal':assert config['evaluation_steps']==[0,*range(64,634,64),634] and 'checkpoint_interval' not in config


@pytest.mark.parametrize('fault',['reward','config','runner','source'])
def test_rehash_cannot_hide_mutation(tmp_path,monkeypatch,fault):
    c=fixture(tmp_path,monkeypatch);module.prepare(c,{})
    root=Path(c['output']);m=module.read(root/'PREPARED.json')
    names={'reward':'objective-extension/extensions/stage2_objective.py','config':'config.json','runner':'run.py','source':'source/forward.py'}
    p=root/names[fault];p.write_text('changed')
    m['files'][names[fault]]=module.file_hash(p);write(root/'PREPARED.json',m);c['manifest_sha256']=module.file_hash(root/'PREPARED.json')
    with pytest.raises(ValueError,match='reconstruction'):module.audit(c,{})


def test_tail_generated_trainer_has_plugin_and_real_tail_selection():
    from tools.reports.context_plugin_trainer import generate
    from tools.reports.context_cached_tail_trainer import generate as tail
    from types import FunctionType
    fn=FunctionType(generate.__code__,dict(generate.__globals__,original_generate=lambda source_root,formal=False:tail(source_root)))
    fn.__kwdefaults__=generate.__kwdefaults__
    source=fn(Path(__file__).parents[1]);compile(source,'<tail-plugin>','exec')
    assert "batches=select_tail(batches,c,population.keys())" in source
    assert "load_dual_actor(c['source_root'],c['objective_plugin'])" in source
    assert "contract['objective_plugin']=dual_actor.objective_plugin_identity" in source


@pytest.mark.parametrize('kind',['tail','efficiency','formal'])
def test_shared_output_contract_includes_objective(tmp_path,monkeypatch,kind):
    # Actual model/reference audit remains the original function; fixture only
    # tests contract wiring, not real output admission.
    c=fixture(tmp_path,monkeypatch);c['kind']=kind;module.prepare(c,{})
    root=Path(c['output']);config=module.read(root/'config.json')
    monkeypatch.setattr(operations.outputs,'expected_contract',lambda *a:{'original':True})
    if kind=='formal':monkeypatch.setattr(operations.formal,'contract',lambda c:{'formal':True})
    def tail(package,collection,*,reference_root):
        return dict(contract=expected_contract(json.loads((Path(package)/'config.json').read_text()),[]))
    def other(package,collection,*,reference_root):
        return dict(contract=outputs.expected_contract(json.loads((Path(package)/'config.json').read_text()),[]))
    target=operations.outputs if kind=='tail' else operations.efficiency if kind=='efficiency' else operations.formal
    monkeypatch.setattr(target,'audit_outputs',tail if kind=='tail' else other)
    result=operations.audit_outputs(root,'fixture',reference_root='fixture')
    assert result['contract']['objective_plugin']==result['objective_plugin']
    assert result['contract']['original']


def test_cannot_launch_without_explicit_stage2_authorization():
    with pytest.raises(ValueError,match='Stage2'):operations.launch({}, {})
