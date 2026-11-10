import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
import watch_omni_forward_acceptance as tool


@pytest.mark.parametrize('failure',[None,'original','create','start','verifier','binding'])
def test_acceptance_is_once_and_never_restarts_original(tmp_path,monkeypatch,failure):
    original='a'*64;verifier='b'*64;calls=[]
    plan=dict(container_id=original,image='image',argv_prefix=['docker','create'])
    monkeypatch.setattr(tool,'check',lambda *a:plan)
    state=dict(Id=original,Image='image',State=dict(Status='exited',ExitCode=1 if failure=='original' else 0))
    def observe(root,cid,image,seconds):
        if cid==original:return state
        return dict(Id=cid,Image=image,State=dict(Status='exited',ExitCode=1 if failure=='verifier' else 0))
    monkeypatch.setattr(tool,'observe',observe)
    monkeypatch.setattr(tool,'inspect',lambda cid:state)
    def remote(argv):
        calls.append(argv[:2])
        if argv[1]=='create':return 'ambiguous' if failure=='create' else verifier
        if argv[1]=='start':
            assert argv[-1]==verifier
            if failure=='start':raise RuntimeError('connection lost after start')
            tool.write(tmp_path/'acceptance.json',dict(container_id='wrong' if failure=='binding' else original,
                forwards=6,weight_updates=0,inspection_sha256=tool.sha(tmp_path/'inspection.json')))
        return ''
    monkeypatch.setattr(tool,'remote',remote)
    if failure is None:
        tool.execute(tmp_path,'pin')
        assert tool.read(tmp_path/'observer-result.json')['forwards']==6
    else:
        with pytest.raises((RuntimeError,ValueError)):tool.execute(tmp_path,'pin')
    if failure=='original':assert not calls and not (tmp_path/'verifier-intent.json').exists()
    else:
        with pytest.raises(FileExistsError):tool.execute(tmp_path,'pin')
        assert calls.count(['docker','create'])==1


def test_transient_inspection_failure_repolls_same_handle(tmp_path,monkeypatch):
    calls=[]
    def inspect(cid):
        calls.append(cid)
        if len(calls)==1:raise tool.subprocess.TimeoutExpired('ssh',20)
        return dict(Id=cid,Image='image',State=dict(Status='exited',ExitCode=0))
    monkeypatch.setattr(tool,'inspect',inspect);monkeypatch.setattr(tool.time,'sleep',lambda _:None)
    assert tool.observe(tmp_path,'a'*64,'image',100)['State']['ExitCode']==0
    assert calls==['a'*64,'a'*64]
