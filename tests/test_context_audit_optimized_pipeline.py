import inspect
from types import SimpleNamespace
from tools.reports import context_audit_optimized_pipeline as m
from tools.reports import context_replica_selection_operations as original


def test_explicit_engine_routes_all_selection_audit_entrypoints_without_mutation():
    prep=SimpleNamespace(prepare=lambda *a:None,audit=lambda *a:None)
    engine=SimpleNamespace(preparation=lambda:prep,reconstruct=lambda c:None)
    before=dict(original.__dict__)
    selection,point=m.optimized_selection(engine)
    assert point.validate.__globals__['preparation'] is prep
    assert point.audit.__globals__['preparation'] is prep
    assert point.run.__globals__['audit'] is point.audit
    assert point.run.__globals__['launch'] is point.launch
    assert selection.__globals__['optimized_preparation'] is prep
    assert selection.__globals__['optimized_point'] is point
    assert selection.__globals__['optimized_curve'].collect
    assert all(original.__dict__[k] is v for k,v in before.items())
