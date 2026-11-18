import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from compare_semantic_context_runs import compare_records, compare_runs, sha, IDENTITY, OBSERVATIONS


def records():
    return [dict(audio_id='a',panel_id=str(i),arm='Q_input',
                 **{f:'same' for f in IDENTITY},
                 **{f:'measured' for f in OBSERVATIONS}) for i in range(2)]


def test_order_independent_exact_match_and_nonlength_changes_detected():
    a=records()
    assert compare_records(a,list(reversed(a)))['all_recorded_inputs_equal']
    for field in ('input_ids_sha256','mappings','positions','audio_tokens'):
        b=copy.deepcopy(a);b[1][field]='different'
        r=compare_records(a,b)
        assert not r['all_recorded_inputs_equal'] and r['identical_requests']==1
        assert set(r['differences'][0]['fields'])=={field}


def test_population_or_task_changes_cannot_be_compared_as_runtime_effect():
    a=records()
    for b in (a[:1],a+a[:1]):
        with pytest.raises(ValueError):compare_records(a,b)
    b=copy.deepcopy(a);b[0]['template_sha256']='new task'
    with pytest.raises(ValueError,match='identity'):compare_records(a,b)


def make_run(root, audio, image):
    root.mkdir();(root/'run').mkdir();(root/'source/tools/jev').mkdir(parents=True)
    def save(path,value):path.write_text(json.dumps(value))
    implementation=root/'source/tools/jev/probe_semantic_actor_context.py'
    implementation.write_text('# frozen diagnostic\n')
    template=dict(panel_id='p',arm='Direct',representation='shared_rules_v1',native_layout='canonical',
                  template_sha256='template',expanded_contract_sha256='contract')
    save(root/'templates.jsonl',template)
    row=dict(template,audio_id='a',audio_sha256=sha(audio),audio_samples=16000,tokens=113,audio_tokens=13,
             positions=[[100]],mappings=[{'values':[False,True]}],input_ids_sha256='tokens',
             overflow={str(n):False for n in (8192,16384,32768,65536)})
    save(root/'run/records.jsonl',row)
    config=dict(image=image,fixed_training_image='fixed',model='same model',
                audio=[dict(id='a',audio=str(audio),sha256=sha(audio))],templates=str(root/'templates.jsonl'),
                pins={str(root/'templates.jsonl'):sha(root/'templates.jsonl')})
    save(root/'config.json',config)
    save(root/'run/result.json',dict(image=image,config_sha256=sha(root/'config.json'),
         records_sha256=sha(root/'run/records.jsonl'),implementation_sha256=sha(implementation),
         requests=1,context_accepted=False,training_dispatch_allowed=False,
         fixed_training_image_used=image=='fixed',versions={'processor':'fixture'},processor_source_sha256='processor',
         per_arm={'Direct':dict(requests=1,min_tokens=113,max_tokens=113,overflow=row['overflow'])}))
    state=dict(Status='exited',Running=False,ExitCode=0,OOMKilled=False)
    save(root/'launch.json',dict(config_sha256=sha(root/'config.json'),container_id='container',
         argv=['docker','create',image,'300','python','probe.py']))
    save(root/'container-inspection.json',[dict(Id='container',Image=image,State=state,
         HostConfig=dict(NetworkMode='none',Devices=[]),Config=dict(Entrypoint=['timeout'],Cmd=['300','python','probe.py']))])
    save(root/'terminal.json',dict(container_id='container',container_image=image,container_state=state,
         inspection_sha256=sha(root/'container-inspection.json')))


def test_file_chain_checks_actual_image_terminal_audio_and_hashes(tmp_path):
    audio=tmp_path/'audio';audio.write_bytes(b'fixture, not a real waveform')
    a=tmp_path/'before';b=tmp_path/'after'
    make_run(a,audio,'cpu');make_run(b,audio,'fixed')
    assert compare_runs(a,b)['all_recorded_inputs_equal']
    terminal=b/'terminal.json';original=terminal.read_text();state=json.loads(original)
    for field,value in [('ExitCode',1),('Running',True),('OOMKilled',True)]:
        changed=copy.deepcopy(state);changed['container_state'][field]=value
        terminal.write_text(json.dumps(changed))
        with pytest.raises(ValueError,match='terminal'):compare_runs(a,b)
    changed=copy.deepcopy(state);changed['container_image']='unexpected';terminal.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='image'):compare_runs(a,b)
    terminal.write_text(original)
    audio.write_bytes(b'changed')
    with pytest.raises(ValueError,match='audio file'):compare_runs(a,b)
    audio.write_bytes(b'fixture, not a real waveform')
    p=b/'run/records.jsonl';p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError,match='records'):compare_runs(a,b)


def test_launch_and_raw_container_receipts_are_required(tmp_path):
    audio=tmp_path/'audio';audio.write_bytes(b'fixture')
    a=tmp_path/'before';b=tmp_path/'after'
    make_run(a,audio,'cpu');make_run(b,audio,'fixed')
    launch=b/'launch.json';old=launch.read_text();d=json.loads(old)
    d['config_sha256']='different';launch.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='launched config'):compare_runs(a,b)
    launch.write_text(old)
    inspection=b/'container-inspection.json';d=json.loads(inspection.read_text())
    d[0]['Config']['Cmd']=['python','other.py'];inspection.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='inspection evidence'):compare_runs(a,b)
    terminal=b/'terminal.json';t=json.loads(terminal.read_text());t['inspection_sha256']=sha(inspection)
    terminal.write_text(json.dumps(t))
    with pytest.raises(ValueError,match='command'):compare_runs(a,b)
