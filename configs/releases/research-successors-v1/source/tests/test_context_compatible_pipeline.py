from types import SimpleNamespace

import pytest

from tools.reports import context_compatible_pipeline as compatibility


@pytest.mark.parametrize('module,route',[
    ('tools.reports.run_context_arithmetic_experiment','audit_context_rl_selection_v2'),
    ('tools.reports.run_context_arithmetic_experiment','audit_context_rl_curve_v2'),
    ('tools.reports.run_fsc_arithmetic_experiment','prepare_fsc_context_source_v2'),
])
def test_only_registered_audit_calls_are_redirected(tmp_path,monkeypatch,module,route):
    original_calls=[];wrapped=[]
    def original(*args):original_calls.append(args)
    driver=SimpleNamespace(invoke=original)
    monkeypatch.setattr(compatibility.importlib,'import_module',lambda name:driver)
    monkeypatch.setattr(compatibility.subprocess,'run',lambda argv,**kw:wrapped.append(argv))
    with pytest.raises(RuntimeError):
        with compatibility.compatible_invocations(module):
            driver.invoke(route,['--spec','same-spec'],tmp_path/'audit.log')
            driver.invoke('launch_existing',['--launch'],tmp_path/'launch.log')
            raise RuntimeError('check restore')
    assert driver.invoke is original
    assert original_calls==[('launch_existing',['--launch'],tmp_path/'launch.log')]
    assert wrapped[0][2]=='tools.reports.context_compatible_pipeline'
    assert wrapped[0][-3:]==['--','--spec','same-spec']
    assert wrapped[0][4]==compatibility.DRIVERS[module][route]


def test_direct_compatibility_is_scoped(monkeypatch):
    from multimodal_jev import context_event_metrics
    original_argv=compatibility.sys.argv
    def run(module,run_name):
        assert context_event_metrics.sum([1e16,1.,-1e16])==1.
        assert compatibility.sys.argv==[module,'--spec','fixed']
        raise RuntimeError('failed audit')
    monkeypatch.setattr(compatibility.runpy,'run_module',run)
    with pytest.raises(RuntimeError):
        compatibility.execute('tools.reports.prepare_fsc_arithmetic_source',['--spec','fixed'])
    assert compatibility.sys.argv is original_argv
    assert 'sum' not in vars(context_event_metrics)


def test_unregistered_command_rejected():
    with pytest.raises(ValueError):compatibility.execute('arbitrary.module',[])
