import os
from pathlib import Path
import pytest
from multimodal_jev.audit_content_session import ContentProofSession


def setup(tmp_path):
    code=tmp_path/'code.py';code.write_text('code')
    data=tmp_path/'data';data.mkdir();(data/'a').write_text('original')
    return ContentProofSession([code],io_workers=2),code,data


def test_full_proof_once_rehashes_and_isolates_result(tmp_path):
    session,code,data=setup(tmp_path);calls=[]
    def compute():
        calls.append(1)
        return {'rows':[p.read_text() for p in data.iterdir()]}
    a=session.get('source',compute);a['rows'].clear()
    assert session.get('source',compute)=={'rows':['original']}
    assert len(calls)==1 and [v['cache_hit'] for v in session.events]==[False,True]


@pytest.mark.parametrize('fault',['content','code','new','delete','symlink','same_stat'])
def test_changed_dependencies_reject_reuse(tmp_path,fault):
    session,code,data=setup(tmp_path)
    session.get('source',lambda: [p.read_text() for p in data.iterdir()])
    p=data/'a';st=p.stat()
    if fault=='content':p.write_text('changed')
    if fault=='code':code.write_text('changed')
    if fault=='new':(data/'b').write_text('added')
    if fault=='delete':p.unlink()
    if fault=='symlink':p.unlink();p.symlink_to(code)
    if fault=='same_stat':p.write_text('mutated!');os.utime(p,ns=(st.st_atime_ns,st.st_mtime_ns))
    with pytest.raises((ValueError,FileNotFoundError)):session.get('source',lambda:pytest.fail('cannot silently recompute frozen data'))


def test_failure_not_published(tmp_path):
    session,_,_=setup(tmp_path)
    def failure():raise ValueError('rejected')
    with pytest.raises(ValueError):session.get('node',failure)
    assert not session.entries


def test_temporary_reconstruction_is_not_external_dependency(tmp_path):
    session,_,data=setup(tmp_path)
    def compute():
        original=(data/'a').read_text()
        p=tmp_path/'temporary';p.write_text(original);value=p.read_text();p.unlink()
        return value
    assert session.get('source',compute)==session.get('source',compute)=='original'


def test_auditor_cannot_reclassify_read_input_as_temporary(tmp_path):
    session,_,data=setup(tmp_path)
    def compute():
        p=data/'a';value=p.read_text();p.write_text('changed');return value
    with pytest.raises(ValueError,match='write an input'):session.get('source',compute)
    assert not session.entries
