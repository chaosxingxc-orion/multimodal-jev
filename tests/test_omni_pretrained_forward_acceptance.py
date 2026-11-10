from copy import deepcopy
import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
from verify_omni_pretrained_forward import check_records,check_container
from probe_omni_pretrained_forward import loading_report


def test_loading_report_preserves_sets_and_mismatch_shapes():
    value=dict(missing_keys=set(),unexpected_keys={'talker.b','talker.a'},
               mismatched_keys=[('name',(2,3),(3,2))],error_msgs=[])
    assert loading_report(value)==dict(missing_keys=[],unexpected_keys=['talker.a','talker.b'],
        mismatched_keys=[['name',[2,3],[3,2]]],error_msgs=[])
    assert isinstance(value['unexpected_keys'],set)
    with pytest.raises(TypeError):loading_report(object())


@pytest.mark.parametrize('change',[None,'running','failed','oom','device','privileged','network','memory','cpu','image','cid','source','time','retry'])
def test_only_exact_successful_cpu_container_is_accepted(change):
    launch=dict(container_id='a'*64,image='pinned',source_commit='commit')
    intent=dict(image='pinned',source_commit='commit',wall_seconds=14400,automatic_retry=False)
    state=dict(Id='a'*64,Image='pinned',State=dict(Status='exited',ExitCode=0,OOMKilled=False),
               HostConfig=dict(Devices=[],Privileged=False,NetworkMode='none',Memory=160*1024**3,NanoCpus=16000000000))
    if change=='running':state['State']['Status']='running'
    if change=='failed':state['State']['ExitCode']=1
    if change=='oom':state['State']['OOMKilled']=True
    if change=='device':state['HostConfig']['Devices']=[{'PathOnHost':'/dev/davinci6'}]
    if change=='privileged':state['HostConfig']['Privileged']=True
    if change=='network':state['HostConfig']['NetworkMode']='host'
    if change=='memory':state['HostConfig']['Memory']=0
    if change=='cpu':state['HostConfig']['NanoCpus']=0
    if change=='image':state['Image']='other'
    if change=='cid':state['Id']='b'*64
    if change=='source':launch['source_commit']='other'
    if change=='time':intent['wall_seconds']=86400
    if change=='retry':intent['automatic_retry']=True
    if change is None:check_container(state,launch,intent)
    else:
        with pytest.raises(ValueError):check_container(state,launch,intent)


def fixture():
    rows=[dict(id=str(i),audio_sha256=['a']*(2 if i in (3,5) else 1),
               input=dict(decisions=[dict(id='f0.present'),dict(id='f0.absent')])) for i in range(6)]
    checks={(r['id'],'f0.present'):dict(input_tokens=10+i) for i,r in enumerate(rows)}
    records=[dict(example_id=r['id'],decision_id='f0.present',audio_count=len(r['audio_sha256']),
                  prompt_tokens=10+i,response_token_ids=[4,5],response_log_probs=[-.4,-.1],
                  seconds=2.,peak_rss_kib=123) for i,r in enumerate(rows)]
    return rows,checks,records


def test_complete_real_input_contract():
    rows,checks,records=fixture()
    check_records(records,rows,checks,[4,5])


@pytest.mark.parametrize('kind',['missing','duplicate','order','decision','audio','prompt','token',
                               'short','nan','inf','positive','boolean','time','rss'])
def test_partial_misbound_or_invalid_forward_rejected(kind):
    rows,checks,records=fixture()
    if kind=='missing':records.pop()
    elif kind=='duplicate':records[-1]=deepcopy(records[0])
    elif kind=='order':records.reverse()
    elif kind=='decision':records[0]['decision_id']='f0.absent'
    elif kind=='audio':records[3]['audio_count']=1
    elif kind=='prompt':records[0]['prompt_tokens']=12
    elif kind=='token':records[0]['response_token_ids']=[4,6]
    elif kind=='short':records[0]['response_log_probs'].pop()
    elif kind in ('nan','inf','positive','boolean'):
        records[0]['response_log_probs'][0]={'nan':float('nan'),'inf':float('-inf'),'positive':.1,'boolean':False}[kind]
    elif kind=='time':records[0]['seconds']=-1
    elif kind=='rss':records[0]['peak_rss_kib']=False
    with pytest.raises(ValueError):check_records(records,rows,checks,[4,5])
