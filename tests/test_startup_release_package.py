import json
from pathlib import Path
from types import SimpleNamespace
from tools.reports.context_audit_release import graph
from test_objective_training_candidate import setup,write
from tools.reports.prepare_context_replica_training_candidate import RUN_PY,ENGINEERING_GATE


def test_release_package_audits_and_preserves_forward(tmp_path,monkeypatch):
    c,parent=setup(tmp_path,monkeypatch)
    original=RUN_PY.replace('{SERVICES}',"[('a','2'),('b','3')]").replace('{world}','2').replace('{train_visible}',"'0,1'").replace('{entry}','context_cached_reference_trainer.py').replace('{GATE}',ENGINEERING_GATE)
    (parent/'run.py').write_text(original)
    from multimodal_jev.baseline import file_hash
    m=json.loads((parent/'PREPARED.json').read_text());m['files']['run.py']=file_hash(parent/'run.py');write(parent/'PREPARED.json',m)
    c['parent']['manifest_sha256']=file_hash(parent/'PREPARED.json')
    p=graph(SimpleNamespace(reconstruct=lambda *a:None,replay=lambda *a:None))['tools.reports.prepare_objective_training_candidate']
    c['manifest_sha256']=p.prepare(c,{})['manifest_sha256'];p.audit(c,{})
    root=Path(c['output']);config=json.loads((root/'config.json').read_text());m=json.loads((root/'PREPARED.json').read_text())
    assert (root/'RUNTIME-SOURCES.json').read_bytes()==(parent/'RUNTIME-SOURCES.json').read_bytes()
    assert config['pins'][c['remote_root']+'/runtime-startup.py']==file_hash(root/'runtime-startup.py')
    assert 'runtime-startup.py' in m['layout']['configs']
    compile((root/'run.py').read_text(),'runner','exec')
