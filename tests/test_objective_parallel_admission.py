from types import ModuleType,SimpleNamespace
import pytest
from tools.reports.objective_parallel_admission import clone_graph,implementation


def test_scoped_graph_propagates_imports_without_global_changes():
    a=ModuleType('fixture_a');exec('def check(x, *, k=2):\n return x*k\n',a.__dict__)
    b=ModuleType('fixture_b');b.a=a;b.imported=a.check
    exec('def run(x):\n return a.check(x)+imported(x)\n',b.__dict__)
    graph=clone_graph([a,b],{('fixture_a','check'):lambda x,**kw:x*3})
    assert graph['fixture_b'].run(4)==24 and b.run(4)==16
    assert graph['fixture_b'].a is graph['fixture_a']


def test_original_exceptions_preserved():
    a=ModuleType('fixture_a');exec('def check():\n raise ValueError("source rejected")\n',a.__dict__)
    with pytest.raises(ValueError,match='source rejected'):clone_graph([a],{})['fixture_a'].check()


def test_real_lifecycle_scopes_all_source_paths():
    from tools.reports import run_objective_acceptance as original
    from tools.reports import context_infonce_selection as source
    f=lambda *a:None
    sequence=implementation(SimpleNamespace(reconstruct=f))
    prefix=sequence.run.__globals__['prefix']
    package=prefix.admission.__globals__['audit_package']
    expected=package.__globals__['expected'];parent=expected.__globals__['parent_api']
    inputs=parent.audit.__globals__['inputs'];runtime=inputs.__globals__['audit_parent']
    source_audit=runtime.__globals__['audit_source']
    assert source_audit.__globals__['reconstruct_plan'] is f
    assert source.reconstruct_plan is not f
    assert original.run is not sequence.run
    # Trace/bootstrap/reference lifecycle predicates remain original bytecode.
    assert prefix.bootstrap_evidence.__code__ is original.prefix.bootstrap_evidence.__code__
    reference=prefix.cached.admission.__globals__['audit_reference']
    gate=reference.__globals__['admission']
    parity=gate.__globals__['audit_parity'].__globals__['audit_package']
    assert parity.__globals__['audit_parent'] is runtime
