import copy
import json
from pathlib import Path
import pytest

from tools.reports.finalize_context_attribute_inputs_v2 import verify_terminal_binding,finish


def files(tmp_path):
    state=dict(Status='exited',ExitCode=0,OOMKilled=False)
    launch=dict(container_id='original',image='fixed',source_manifest_sha256='a'*64)
    transport=dict(container_id='original',image='fixed',state=state,source_manifest_sha256='a'*64,
        terminal=dict(exit_code=0,completed=40513))
    inspect=dict(Id='original',Image='fixed',State=state,HostConfig=dict(Devices=[],NetworkMode='none'))
    for name,value in [('transport',transport),('launch',launch),('inspect.json',inspect)]:
        (tmp_path/name).write_text(json.dumps(value))
    return transport,inspect


def test_exact_owner_terminal_required(tmp_path):
    files(tmp_path)
    result=verify_terminal_binding(tmp_path/'transport',tmp_path/'launch',tmp_path)
    assert result['terminal']['completed']==40513


@pytest.mark.parametrize('fault',['running','exit','oom','partial','different_job','different_image','device','network','manifest'])
def test_incomplete_or_changed_terminal_rejected(tmp_path,fault):
    t,i=files(tmp_path)
    if fault=='running':t['state']['Status']='running'
    if fault=='exit':t['state']['ExitCode']=1
    if fault=='oom':t['state']['OOMKilled']=True
    if fault=='partial':t['terminal']['completed']=3
    if fault=='different_job':i['Id']='replacement'
    if fault=='different_image':i['Image']='new'
    if fault=='device':i['HostConfig']['Devices']=['npu']
    if fault=='network':i['HostConfig']['NetworkMode']='host'
    if fault=='manifest':t['source_manifest_sha256']='bad'
    (tmp_path/'transport').write_text(json.dumps(t));(tmp_path/'inspect.json').write_text(json.dumps(i))
    with pytest.raises(ValueError):verify_terminal_binding(tmp_path/'transport',tmp_path/'launch',tmp_path)


def test_missing_terminal_never_publishes(tmp_path):
    args={k:tmp_path/'missing' for k in ['collection','package','transport','launch','anchors','dataset',
            'original_schedule','base_config']}
    with pytest.raises(FileNotFoundError):finish(source_root='.',original_sha256='a'*64,output=tmp_path/'out',**args)
    assert not (tmp_path/'out').exists()
