import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
from multimodal_jev.consumption import canonical,request_from_example
from multimodal_jev.task_schema import Candidate,DecisionSchema,JevExample

spec=importlib.util.spec_from_file_location('teacher_prepare',Path(__file__).parents[1]/'tools/jev/prepare_teacher_feedback.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def setup(tmp_path,monkeypatch):
    root=tmp_path/'learning';run=root/'run';run.mkdir(parents=True);(run/'transcriptions').mkdir()
    schema=DecisionSchema('task','choice','Choose',(Candidate('a','on'),Candidate('b','off')))
    requests=[];targets=[];predictions=[]
    for i in range(64):
        ex=JevExample(str(i),'train','audio',schema,'a',{'source_group':str(i),'audio_sha256':'hash'+str(i)})
        request=request_from_example(ex);digest=hashlib.sha256(canonical(request)).hexdigest()
        requests.append(request);targets.append({'id':str(i),'target':'a','request_sha256':digest})
        (run/'transcriptions'/('hash'+str(i)+'.json')).write_text(json.dumps({'audio_sha256':'hash'+str(i),'transcription':'turn on'}))
        for j in range(4):
            predictions.append({'example_id':str(i),'replicate':j,'request_sha256':digest,'raw_text':json.dumps({'value':'a' if j==0 else 'b'})})
    req=tmp_path/'requests.jsonl';req.write_bytes(b''.join(canonical(r) for r in requests))
    gold=tmp_path/'targets.jsonl';gold.write_bytes(b''.join(canonical(r) for r in targets))
    (run/'predictions.jsonl').write_bytes(b''.join(canonical(r) for r in predictions))
    prompt=tmp_path/'prompt.txt';prompt.write_text('classify')
    cfg={'split':'train','purpose':'tfrl-learning','sampling':{'temperature':.7,'n':4},
         'parameters_frozen':True,'source_groups':[str(i) for i in range(64)],'requests':str(req),'instruction':str(prompt)}
    identity={'config':cfg,'requests_sha256':module.file_hash(req),'instruction_sha256':module.file_hash(prompt),'models':{'audio_model':{'policy_id':'base'}}}
    (run/'identity.json').write_text(json.dumps(identity))
    (run/'terminal.json').write_text('{"exit_code":0}')
    (run/'result.json').write_text(json.dumps({'status':'passed','weight_updates':0,'predictions_sha256':module.file_hash(run/'predictions.jsonl')}))
    (root/'learning-selection-split.json').write_text(json.dumps({'learning':cfg['source_groups'],'selection':['heldout']}))
    executor=tmp_path/'executor.py';executor.write_text('raise AssertionError("must not execute")')
    monkeypatch.setattr(module,'EXECUTOR_SHA',module.file_hash(executor))
    teacher=tmp_path/'teacher.json';teacher.write_text(json.dumps({'endpoint':'https://example.invalid/chat/completions','model':'glm-5.3-flash','api_key_env':'TEST_UNUSED_KEY'}))
    return run,gold,tmp_path/'output',executor,teacher


def test_preparation_preserves_real_mixed_groups_without_dispatch(tmp_path,monkeypatch):
    args=setup(tmp_path,monkeypatch);_,receipt=module.prepare(*args)
    assert receipt['draws']==256 and receipt['representatives']==8 and receipt['API_sent'] is False
    messages=json.loads((args[2]/'messages.private.json').read_text());context=json.loads(messages[1]['content'])
    assert context['selection_or_test_feedback'] is False
    assert 'heldout' not in json.dumps(context)
    assert sum(r['correct'] for r in context['representatives'][0]['group_outputs'])==1
    assert not (args[2]/'response').exists()


def test_test_split_and_selection_overlap_rejected(tmp_path,monkeypatch):
    args=setup(tmp_path,monkeypatch);path=args[0]/'identity.json';identity=json.loads(path.read_text())
    identity['config']['split']='test';path.write_text(json.dumps(identity))
    with pytest.raises(ValueError,match='TRAIN'):module.prepare(*args)
    identity['config']['split']='train';path.write_text(json.dumps(identity))
    split=args[0].parent/'learning-selection-split.json';data=json.loads(split.read_text());data['selection']=['0'];split.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='overlap'):module.prepare(*args)
