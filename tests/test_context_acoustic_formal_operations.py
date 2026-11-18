from pathlib import Path
import pytest
from test_context_macro_training_operations import fixture
from tools.reports import context_acoustic_formal_operations as a
from tools.reports import context_macro_formal_operations as f


@pytest.mark.parametrize('role',['client','preflight','service-0','service-1'])
def test_formal_actor_stays_same_and_service_keeps_acoustic_entry(tmp_path,role):
    c,_=fixture(tmp_path)
    before=f.command(c,role);after=a.command(c,role)
    if role.startswith('service-'):
        assert len(before)==len(after)
        assert [(x,y) for x,y in zip(before,after) if x!=y]==[
            ('tools.reports.context_dual_runtime_candidate','tools.reports.context_acoustic_service_candidate')]
    else:assert before==after
    assert a._validate.__globals__['command'] is a.command
    assert f.validate.__globals__['command'] is f.command


def test_validation_rejects_changed_acoustic_source(tmp_path,monkeypatch):
    project=tmp_path/'project';package=tmp_path/'prepared/package'
    for name in a.acoustic.ADDITIONS:
        for root in (project,package):
            path=root/'tools/reports'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('sealed')
    c=dict(project=str(project),prepared=str(package.parent))
    monkeypatch.setattr(a,'_validate',lambda c:{'formal_launch_admitted':False})
    assert a.validate(c)=={'formal_launch_admitted':False}
    (package/'tools/reports'/a.acoustic.ADDITIONS[0]).write_text('changed')
    with pytest.raises(ValueError,match='acoustic implementation'):a.validate(c)
