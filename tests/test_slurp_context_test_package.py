"""Package binding checks with source/data verification substitutes."""
import json
from pathlib import Path
import pytest
from tools.reports import prepare_slurp_context_test_package as package
from tools.reports import launch_slurp_context_test_pair as launcher


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    source=tmp_path/'frozen';source.mkdir();(source/'PACKAGE.json').write_text('{}')
    (source/'source-selection.json').write_text('{}')
    inputs=tmp_path/'inputs';inputs.mkdir();(inputs/'MANIFEST.json').write_text('{}')
    (inputs/'inputs.jsonl').write_text('{}\n');(inputs/'definitions.json').write_text('{}')
    processor=tmp_path/'processor';processor.mkdir()
    for name in ('MANIFEST.json','inputs.jsonl','audit.json'):(processor/name).write_text('{}')
    model=tmp_path/'model';model.mkdir();(model/'model.safetensors').write_bytes(b'fixture')
    monkeypatch.setattr(package,'MODEL',str(model))
    monkeypatch.setattr(package,'verify_source',lambda *a:{})
    seen=[]
    monkeypatch.setattr(package,'configs',lambda c:seen.append(c))
    from multimodal_jev import local_runtime_storage
    monkeypatch.setattr(local_runtime_storage,'check_paths',lambda p:None)
    monkeypatch.chdir(tmp_path)
    for name in ('src/model.py','tools/jev/helper.py','tools/reports/slurp_context_test_runtime.py',
                 'tools/reports/preflight_slurp_context_test_inputs.py','tools/reports/run_slurp_context_test_pair.py'):
        p=Path(name);p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# fixture\n')
    out=tmp_path/'runtime';spec=dict(source_package=str(source),source_manifest_sha256=package.sha(source/'PACKAGE.json'),
        inputs=str(inputs),processor_receipts_root=str(processor),processor_audit=str(processor/'audit.json'))
    package.prepare(spec,out,package.PREFIX+'slurp-context-test-pair-fixture')
    return out,package.sha(out/'PACKAGE.json'),seen


def test_copies_receipts_and_checks_relocated_local_inputs(prepared):
    out,digest,seen=prepared;m=package.verify(out,digest);c=package.read(out/'config.json')
    assert seen[-1]['source_package']==str(out/'frozen-source')
    assert seen[-1]['inputs']['processor_receipts']==str(out/'processor/inputs.jsonl')
    assert seen[-1]['inputs']['cache_root']==package.CACHE
    argv=launcher.command(m,c)
    mounts=[argv[i+1] for i,v in enumerate(argv) if v=='-v']
    assert package.CACHE+':'+package.CACHE+':ro' in mounts
    assert m['remote_root']+'/processor:'+m['remote_root']+'/processor:ro' in mounts
    assert m['remote_root']+'/source/tools/reports/run_slurp_context_test_pair.py' in argv


@pytest.mark.parametrize('fault',['processor','source','extra','config','model','symlink'])
def test_runtime_corruption_rejected(prepared,fault):
    out,digest,_=prepared
    if fault=='model':Path(package.MODEL,'model.safetensors').write_bytes(b'changed')
    elif fault=='symlink':(out/'link').symlink_to(out/'config.json')
    else:
        name={'processor':'processor/inputs.jsonl','source':'source/src/model.py','extra':'extra','config':'config.json'}[fault]
        (out/name).write_text('changed')
    with pytest.raises(ValueError):package.verify(out,digest)


def test_default_launcher_never_calls_remote(prepared,monkeypatch,capsys):
    out,digest,_=prepared
    import sys
    monkeypatch.setattr(sys,'argv',['launch','--package',str(out),'--manifest-sha256',digest])
    def reject(*a,**k):raise AssertionError('no remote')
    monkeypatch.setattr(launcher.subprocess,'run',reject)
    monkeypatch.setattr(launcher.subprocess,'check_output',reject)
    launcher.main()
    assert json.loads(capsys.readouterr().out)['status']=='validated_only_no_remote_calls'
