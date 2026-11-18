from types import SimpleNamespace
from tools.reports import context_stage1_priority_pipeline as m


def test_stage1_finishes_before_efficiency_and_preserves_downstream():
    fn=m.adapted_run(lambda *a:None)
    names=fn.__code__.co_names
    assert 'ThreadPoolExecutor' not in names
    assert all(n in names for n in ('finite_and_efficiency','formal_training','transfers'))
    assert fn.__globals__['retire'] is m.retire
    assert fn.__globals__['validate'] is m.validate


def test_isolated_engine_routes_selection_preparation_and_reconstruction():
    prep=SimpleNamespace(prepare=lambda *a:None,audit=lambda *a:None)
    engine=SimpleNamespace(preparation=lambda:prep,reconstruct=lambda *a:None)
    selection,point=m.optimized_selection(engine)
    assert point.validate.__globals__['preparation'] is prep
    assert point.audit.__globals__['preparation'] is prep
    assert selection.__globals__['optimized_preparation'] is prep


def test_actual_phase_order_keeps_complete_downstream(tmp_path, monkeypatch):
    from tools.reports import context_cached_grpo_source_plan as source
    import multimodal_jev.local_checkpoint as checkpoint
    monkeypatch.setattr(checkpoint,'check_paths',lambda *a:None)
    events=[]
    def selection(*args,**kwargs):
        events.append('grpo_curve' if kwargs.get('grpo') else 'stage1_curve')
        return {},{}
    fn=m.adapted_run(selection)
    scope=fn.__globals__
    scope['validate']=lambda c:{'selection':{'point_template':{'selection_inputs':{}}}}
    scope['retire']=lambda *a:events.append('retire')
    scope['finite_and_efficiency']=lambda *a:events.append('efficiency')
    def formal(*a):
        events.append('formal')
        return {},{'training_package':{'parent':{}}},None
    scope['formal_training']=formal
    scope['transfers']=lambda *a:events.append('transfer') or {}
    monkeypatch.setattr(source,'build',lambda *a:{})
    fn({'output':str(tmp_path/'run')},{})
    assert events==['retire','stage1_curve','efficiency','formal','grpo_curve','transfer']


def test_adopted_endpoint_skips_prepare_and_launch(tmp_path,monkeypatch):
    import multimodal_jev.local_checkpoint as checkpoint
    monkeypatch.setattr(checkpoint,'check_paths',lambda *a:None)
    prep=SimpleNamespace(prepare=lambda *a:(_ for _ in ()).throw(AssertionError('unexpected prepare')),
                         audit=lambda *a:(_ for _ in ()).throw(AssertionError('unexpected audit')))
    engine=SimpleNamespace(preparation=lambda:prep,reconstruct=lambda *a:None)
    fn=m.adopted_selection(engine);g=fn.__globals__;events=[]
    def adopt(*args):events.append('adopt');return dict(step=318,audit_config='original',audit_config_sha256='hash')
    g['adopt_point']=adopt
    g['optimized_curve']=SimpleNamespace(collect=lambda c,p:dict(status='curve',evaluations=c['evaluations']))
    c=dict(output=str(tmp_path),evaluation_workers=2,adopt_selection={'bound':True})
    cc,result=fn(c,{},[318],{},None)
    assert events==['adopt'] and result['evaluations'][0]['audit_config']=='original'


def test_adopt_wait_collect_audit_order_without_model_dispatch(tmp_path,monkeypatch):
    import json
    import multimodal_jev.local_checkpoint as checkpoint
    import tools.reports.context_replica_training_operations as lifecycle
    import tools.jev.collect_context_replica_training as transport
    import tools.reports.context_dual_probe_operations as runtime
    from contextlib import nullcontext
    monkeypatch.setattr(checkpoint,'check_paths',lambda *a:None)
    lease=tmp_path/'lease';lease.touch()
    ops=dict(host='host',lease=str(lease),selection_package={'source_step':318})
    handle=dict(container_id='a'*64,argv=['original'],image='image')
    monkeypatch.setattr(m.original,'checked_source',lambda b:ops if b=='ops' else handle)
    events=[]
    monkeypatch.setattr(lifecycle,'wait_exact',lambda remote,cid:events.append('wait') or 0)
    def remote(argv,**kw):
        assert 'docker inspect' in argv[-1]
        events.append('inspect')
        return SimpleNamespace(stdout=json.dumps([{'Id':'a'*64}]),check_returncode=lambda:None)
    monkeypatch.setattr(m.subprocess,'run',remote)
    monkeypatch.setattr(transport,'collect',lambda *a,**kw:events.append('collect'))
    monkeypatch.setattr(runtime,'check_runtime',lambda *a:events.append('runtime'))
    def audit(bound,p):
        events.append('audit')
        assert bound['launch']=='original-launch'
        return dict(status='complete_replica_infonce_selection_audited',source_step=318)
    c=dict(operations_root=str(tmp_path/'ops'),adopt_selection=dict(operations='ops',launch={'path':'original-launch'}))
    entry=m.adopt_point(c,{},SimpleNamespace(slot=lambda *a:nullcontext()),SimpleNamespace(audit=audit))
    assert events==['wait','inspect','collect','runtime','audit'] and entry['step']==318
