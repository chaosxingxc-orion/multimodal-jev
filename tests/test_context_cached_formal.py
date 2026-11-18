import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tools.reports import context_cached_formal as module
from test_prepare_context_cached_training import write,reseal,setup
from test_context_replica_formal_candidate import batch_config,batches


def fixture(tmp_path,monkeypatch):
    spec,config,_,_=setup(tmp_path,monkeypatch)
    config.update(batch_config=dict(batch_config(2),inference_batch_queries=32),microbatch=8,
        global_batch=128,stage='task',seed=317)
    write(Path(spec['parent']['output'])/'config.json',config)
    efficiency=dict(training_package=dict(parent=spec['parent'],reference_job=spec['reference_job']))
    monkeypatch.setattr(module.efficiency,'audit',lambda *a:dict(status='cached_100step_efficiency_outputs_audited',
        consumed_reference_queries=12800,online_reference_calls=0))
    spec.update(acceptance_kind='formal_source',efficiency_audit=efficiency,evaluation_steps=list(module.STEPS))
    return spec


def test_formal_package_preserves_runtime_and_endpoint_and_uses_full_curve(tmp_path,monkeypatch):
    c=fixture(tmp_path,monkeypatch);parent=Path(c['parent']['output'])
    before={str(p):module.file_hash(p) for p in parent.rglob('*') if p.is_file()}
    c['manifest_sha256']=module.package_action(c,{})['manifest_sha256']
    assert before=={str(p):module.file_hash(p) for p in parent.rglob('*') if p.is_file()}
    proof=module.package_action(c,{},audit=True)
    assert proof['actor_updates']==634 and not proof['formal_dispatch_admitted'] and not proof['npu_executed']
    root=Path(c['output']);config=module.read(root/'config.json')
    assert config['evaluation_steps']==module.STEPS and config['stop_update']==634 and config['start_update']==0
    assert 'checkpoint_interval' not in config and 'resume_root' not in config
    assert config['source_policy_id']==module.read(parent/'config.json')['source_policy_id']
    assert (root/'RUNTIME-SOURCES.json').read_bytes()==(parent/'RUNTIME-SOURCES.json').read_bytes()
    calls=[]
    monkeypatch.setitem(sys.modules,'context_process_supervisor',SimpleNamespace(
        supervise_services=lambda **kw:(calls.append(kw) or {'passed':True})))
    from multimodal_jev import local_runtime_storage
    monkeypatch.setattr(local_runtime_storage,'require_training_storage',lambda _:None)
    script=(root/'run.py').read_text().replace("Path('/jev-tmp')",'Path('+repr(str(tmp_path))+')')
    with pytest.raises(SystemExit) as e:exec(compile(script,'<formal-runner>','exec'),{'__file__':str(root/'run.py')})
    assert e.value.code==0 and len(calls)==1
    assert calls[0]['actor_env']['ASCEND_RT_VISIBLE_DEVICES']=='0,1'
    assert [s['env']['ASCEND_RT_VISIBLE_DEVICES'] for s in calls[0]['services']]==['2','3']
    config['start_update']=100;write(root/'config.json',config)
    with pytest.raises(ValueError):exec(compile(script,'<formal-runner>','exec'),{'__file__':str(root/'run.py')})
    assert len(calls)==1


@pytest.mark.parametrize('fault',['curve','endpoint','reference','efficiency','batch','partial'])
def test_unrelated_or_incomplete_evidence_cannot_prepare_formal(tmp_path,monkeypatch,fault):
    c=fixture(tmp_path,monkeypatch)
    if fault=='curve':c['evaluation_steps']=[0,634]
    elif fault=='endpoint':c['parent']=dict(c['parent'],manifest_sha256='efficiency-endpoint')
    elif fault=='reference':c['reference_job']=dict(c['reference_job'],manifest_sha256='other')
    elif fault=='efficiency':monkeypatch.setattr(module.efficiency,'audit',lambda *a:dict(status='partial',consumed_reference_queries=12800,online_reference_calls=0))
    else:
        path=Path(c['parent']['output'])/'config.json';cfg=module.read(path)
        cfg['batch_config']['training_devices' if fault=='batch' else 'source_queries']=4 if fault=='batch' else 200
        write(path,cfg)
    with pytest.raises(ValueError):module.package_action(c,{})
    assert not Path(c['output']).exists()


@pytest.mark.parametrize('fault',['curve','objective','source','cadence','start','runner'])
def test_rehashed_formal_mutations_are_rejected(tmp_path,monkeypatch,fault):
    c=fixture(tmp_path,monkeypatch);module.package_action(c,{})
    root=Path(c['output']);config=module.read(root/'config.json')
    if fault=='curve':config['evaluation_steps']=[0,634]
    elif fault=='objective':config['stage2_objective']['binary_nll_coefficient']=0.
    elif fault=='source':(root/'source/test.py').write_text('changed')
    elif fault=='cadence':config['checkpoint_interval']=1
    elif fault=='start':config['start_update']=100
    else:(root/'run.py').write_text('pass')
    write(root/'config.json',config);reseal(c)
    with pytest.raises(ValueError):module.package_action(c,{},audit=True)


def test_formal_schedule_adapter_passes_derived_plan_and_checks_full_population():
    # Execute the actual adapted auditor through its schedule check. Model and
    # transport proofs are outside this targeted test, never substituted as NPU admission.
    source=module.formal_shared_auditor(dict(module.outputs.shared.audit_outputs.__globals__))
    # Extract the exact compiled schedule call from the adapted function's code
    # via the same fixed substitution, then run against real two-epoch rows.
    import inspect
    tree=ast.parse(inspect.getsource(module.outputs.shared.audit_outputs).replace(
        "validate_schedule(batches, population.keys(), c['batch_config'])",
        'validate_schedule(batches, population.keys(), plan)'))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
        and isinstance(n.value.func,ast.Name) and n.value.func.id=='validate_schedule']
    assert len(calls)==1 and source.__name__=='audit_outputs'
    from tools.reports.context_replica_formal_candidate import validate_schedule
    plan=module.derive(dict(batch_config(2),inference_batch_queries=32));rows=batches(plan)
    population={s['id']:{} for row in rows for s in row['slots']}
    code=compile(ast.Module(body=calls,type_ignores=[]),'<formal-schedule-check>','exec')
    scope=dict(validate_schedule=validate_schedule,batches=rows,population=population,plan=plan)
    exec(code,scope)
    rows[0]['slots'][0]['id']=rows[0]['slots'][1]['id']
    with pytest.raises(ValueError):exec(code,scope)


def test_formal_contract_matches_actual_generated_trainer_assignment(tmp_path,monkeypatch):
    c=fixture(tmp_path,monkeypatch);module.package_action(c,{})
    config=module.read(Path(c['output'])/'config.json')
    monkeypatch.setattr(module.outputs,'expected_contract',lambda *a:{'fixture':'base'})
    from tools.reports.context_cached_reference_trainer import generate
    tree=ast.parse(generate(Path(__file__).resolve().parents[1],formal=True))
    nodes=[n for n in ast.walk(tree) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='contract'
            and isinstance(t.slice,ast.Constant) and t.slice.value=='formal_experiment' for t in n.targets)]
    assert len(nodes)==1
    scope=dict(contract={'fixture':'base'},formal=module.contract(config))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual-formal-contract>','exec'),scope)
    assert module.expected_contract(config,[])==scope['contract']


def test_launch_requires_explicit_formal_flag_and_revalidates_before_remote_work(monkeypatch):
    for c in ({},{'efficiency_dispatch_allowed':True},{'finite_training_dispatch_allowed':True}):
        with pytest.raises(ValueError,match='full source'):module.launch(c,{})
    seen=[]
    def refuse(c):seen.append(c);raise ValueError('efficiency evidence missing')
    monkeypatch.setattr(module,'validate',refuse)
    with pytest.raises(ValueError,match='evidence missing'):
        module.launch({'formal_training_dispatch_allowed':True},{})
    assert len(seen)==1 and seen[0]['formal_training_dispatch_allowed'] is True


@pytest.mark.parametrize('failure',[None,'launch','collection','audit','partial'])
def test_formal_lifecycle_never_confuses_exit_zero_with_audit(tmp_path,monkeypatch,failure):
    from multimodal_jev import local_checkpoint
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda _:None)
    monkeypatch.setattr(module,'check_paths',lambda _:None)
    c=dict(formal_training_dispatch_allowed=True,operations_root=str(tmp_path/'operations'),
        tag='formal-fixture',training_package={'fixture':True},reference_audit={'fixture':True})
    calls=[]
    def launch(c,p):
        calls.append('launch')
        if failure=='launch':raise ValueError('launch failed')
        return dict(status='wrong' if failure=='collection' else 'collected_independent_cached_training_audit_required')
    def audit(c,p):
        calls.append('audit')
        if failure=='audit':raise ValueError('audit failed')
        return dict(status='formal_cached_reference_training_outputs_audited',
            real_query_exposures=81026,updates=[{}]*(633 if failure=='partial' else 634))
    monkeypatch.setattr(module,'launch',launch);monkeypatch.setattr(module,'audit',audit)
    if failure:
        with pytest.raises(ValueError):module.run(c,{})
    else:module.run(c,{})
    terminal=module.read(Path(c['operations_root'])/'finalization/terminal.json')
    assert terminal['exit_code']==int(failure is not None) and terminal['restart'] is False
    assert calls==(['launch'] if failure in ('launch','collection') else ['launch','audit'])
    if not failure:assert terminal['selection_evaluation_pending'] and terminal['transfer_evaluation_pending']
    with pytest.raises(FileExistsError):module.run(c,{})


@pytest.mark.parametrize('step',[1,64,634])
def test_actual_formal_publication_uses_sparse_optimizer_and_every_step_policy(tmp_path,step):
    from tools.reports.context_cached_reference_trainer import generate
    from tools.reports.audit_context_replica_training import published_policy_commit
    tree=ast.parse(generate(Path(__file__).resolve().parents[1],formal=True))
    blocks=[n for n in ast.walk(tree) if isinstance(n,ast.If)
        and any(isinstance(child,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='published' for t in child.targets)
            for child in n.body)]
    assert len(blocks)==1
    scope=dict(rank=0,index=step-1,world=2,root=tmp_path,save_now=step in module.STEPS,
        contract={'fixture':'formal'},joint={'policy_id':f'policy-{step}'},
        file_hash=lambda p:'hash:'+p.name,commit_json=write)
    exec(compile(ast.Module(body=blocks,type_ignores=[]),'<actual-formal-publication>','exec'),scope)
    published=published_policy_commit(tmp_path,step,formal=True)
    assert published['updates']==step and 'rank_shards' not in published
    checkpoint=tmp_path/f'checkpoint-{step:04d}-committed.json'
    assert checkpoint.exists()==(step in module.STEPS)
    if checkpoint.exists():
        saved=module.read(checkpoint)
        assert {k:v for k,v in saved.items() if k!='rank_shards'}==published
        assert len(saved['rank_shards'])==2
