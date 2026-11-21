import json
from pathlib import Path
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.stage1_package_sources import FINITE, TAIL, copy_checked
from tools.reports.context_stage1_start_pipeline import cpu_source, finish_formal


def snapshot(tmp):
    root=tmp/'test-package';root.mkdir()
    files={}
    for n in (*FINITE,*TAIL,'src/multimodal_jev/__init__.py'):
        p=root/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(n);files[n]=file_hash(p)
    m=root/'FILES.json';m.write_text(json.dumps(files))
    return root,dict(collection=str(tmp/'test-collected'),source_manifest_sha256=file_hash(m))


@pytest.mark.parametrize('fault',['extra','link','removed','undeclared_supervisor','changed_after_verify'])
def test_snapshot_exact_runtime_before_packaging(tmp_path,fault):
    root,r=snapshot(tmp_path);name=FINITE[1];m=root/'FILES.json'
    source,files=cpu_source(r,io_workers=2)
    with pytest.raises(ValueError):
        if fault=='changed_after_verify':
            (root/name).write_text('different');copy_checked(source,tmp_path/'out',[name],files);return
        if fault=='extra':(root/'extra.py').touch()
        if fault=='link':(root/'x').symlink_to(root/name)
        if fault in ('removed','undeclared_supervisor'):(root/name).unlink()
        if fault=='undeclared_supervisor':
            f=json.loads(m.read_text());del f[name];m.write_text(json.dumps(f));r['source_manifest_sha256']=file_hash(m)
        cpu_source(r,io_workers=2)


def test_current_cpu_pack_includes_all_stage1_support():
    p=Path(__file__).resolve().parents[1]/'configs/portable/operations/context-stage1-margin-cpu-tests.json'
    assert set((*FINITE,*TAIL))<=set(json.loads(p.read_text())['support_files'])


def test_formal_finish_only_after_matching_independent_collection(tmp_path):
    collected=dict(status='collected_independent_training_audit_required',container_id='cid',exit_code=0)
    proof=dict(status='formal_infonce_artifacts_audited',container_id='other',updates=[{},{}])
    calls=[]
    def audit(c):calls.append(c);return proof
    with pytest.raises(ValueError):finish_formal(tmp_path,dict(collected,exit_code=1),{},audit)
    assert not calls
    with pytest.raises(ValueError):finish_formal(tmp_path,collected,{},audit)
    assert not (tmp_path/'terminal.json').exists()
    proof['container_id']='cid';calls.clear();finish_formal(tmp_path,collected,{},audit)
    assert len(calls)==1 and json.loads((tmp_path/'terminal.json').read_text())['updates']==2
