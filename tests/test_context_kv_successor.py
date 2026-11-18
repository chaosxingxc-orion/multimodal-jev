import json
from pathlib import Path

import pytest

from tools.reports import run_context_kv_successor as s


def setup(tmp_path, monkeypatch, success=True, exact=True):
    def write(p,v): p.write_text(json.dumps(v)); return str(p)
    ops = write(tmp_path/'ops.json',dict(parent_operations=dict(operations_root=str(tmp_path/'acoustic'),tag='a')))
    driver = write(tmp_path/'driver.json', dict(pid=1))
    terminal = write(tmp_path/'terminal.json',dict(exit_code=0 if success else 1,status='acoustic_and_bucket_efficiency_compared'))
    events=[]
    monkeypatch.setattr(s,'pins',lambda c:None)
    monkeypatch.setattr(s,'check_paths',lambda c:None)
    monkeypatch.setattr(s.kv,'validate',lambda c:None)
    monkeypatch.setattr(s,'wait_driver',lambda *a:events.append('wait'))
    monkeypatch.setattr(s.acoustic,'audit',lambda *a:events.append('audit') or dict(exact_probabilities_and_logits=exact,exact_sampling=exact))
    def launch(c,p):
        assert c['finite_inference_dispatch_allowed'] is True
        events.append('launch'); return {'result':'complete'}
    monkeypatch.setattr(s.kv,'namespace',lambda c:dict(launch=launch))
    return dict(output=str(tmp_path/'output'),operations=ops,predecessor_driver=driver,predecessor_terminal=terminal),events


def test_only_launch_after_exact_wait_and_independent_admission(tmp_path,monkeypatch):
    c,e=setup(tmp_path,monkeypatch);s.run(c,{})
    assert e==['wait','audit','launch']
    assert json.loads((Path(c['output'])/'terminal.json').read_text())['training_dispatched'] is False
    assert 'finite_inference_dispatch_allowed' not in json.loads(Path(c['operations']).read_text())


@pytest.mark.parametrize('success,exact',[(False,True),(True,False)])
def test_failed_predecessor_or_numerical_admission_never_dispatches(tmp_path,monkeypatch,success,exact):
    c,e=setup(tmp_path,monkeypatch,success,exact)
    with pytest.raises(ValueError):s.run(c,{})
    assert 'launch' not in e
    assert json.loads((Path(c['output'])/'terminal.json').read_text())['restart'] is False
