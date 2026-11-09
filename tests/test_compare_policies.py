import json
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.compare_policies import bootstrap,compare
from multimodal_jev.evaluate import evaluate
from multimodal_jev.task_schema import JevExample,DecisionSchema


def test_cluster_bootstrap_preserves_pairing():
    rows=[('a',False,True),('a',False,True),('b',True,False)]
    result=bootstrap(rows,2000,17)
    assert result['clusters']==2 and result['count']==3
    assert result['delta']==pytest.approx(1/3)
    assert result['delta_ci95']==[-1,1]
    assert bootstrap([(c,l,l) for c,l,r in rows],2000,17)['delta_ci95']==[0,0]
    assert bootstrap([('a',False,True)],2000,17)['delta_ci95'] is None


def fixture(tmp_path):
    manifest=tmp_path/'manifest'
    examples=[JevExample(str(i),'valid','fixture',DecisionSchema('s','noul','True?'),True,
                        {'speaker_id':f's{i}','source_group':str(i),'variant':'noul.True'}) for i in range(3)]
    manifest.write_text(''.join(json.dumps(ex.to_record())+'\n' for ex in examples))
    runs=[]
    for name,value in [('base',False),('candidate',True)]:
        run=tmp_path/name;run.mkdir();runs.append(run)
        (run/'predictions.jsonl').write_text(''.join(json.dumps({'example_id':ex.example_id,'raw_text':json.dumps({'value':value})})+'\n' for ex in examples))
        (run/'metrics.json').write_text(json.dumps(evaluate(manifest,run/'predictions.jsonl')))
        (run/'identity.json').write_text(json.dumps({'manifest_sha256':file_hash(manifest),
            'config':{'max_new_tokens':64,'device':'cpu','dtype':'float32','seed':17}}))
        (run/'terminal.json').write_text('{"exit_code":0}')
    return manifest,*runs


def test_completed_comparison_and_artifact_gate(tmp_path):
    manifest,base,candidate=fixture(tmp_path)
    result=compare(manifest,base,candidate,100)
    cell=result['comparison']['valid/noul.True']
    assert cell['delta']==1 and cell['delta_ci95']==[1,1]
    (candidate/'predictions.jsonl').write_text('')
    with pytest.raises(ValueError,match='artifacts changed'):compare(manifest,base,candidate,100)


def test_failed_or_different_protocol_evaluation_rejected(tmp_path):
    manifest,base,candidate=fixture(tmp_path)
    identity=json.loads((candidate/'identity.json').read_text());identity['config']['max_new_tokens']=128
    (candidate/'identity.json').write_text(json.dumps(identity))
    with pytest.raises(ValueError,match='protocol differs'):compare(manifest,base,candidate,100)
    (candidate/'terminal.json').write_text('{"exit_code":1}')
    with pytest.raises(ValueError,match='successful evaluation'):compare(manifest,base,candidate,100)


def text_fixture(tmp_path):
    manifest,base,candidate=fixture(tmp_path)
    for run,mode in ((base,'asr-text'),(candidate,'reference-text')):
        path=run/'identity.json';identity=json.loads(path.read_text())
        identity['config'].update(input_mode=mode,enable_thinking=False,cpu_threads=4,
                                  model_id='fixed-text-model',model_revision='revision')
        identity.update(policy_id='same-weights',checkpoint_files={'weights':'hash'},
            source_sha256={name:'source-hash' for name in
                           ('text_baseline.py','audio_policy.py','task_schema.py')})
        path.write_text(json.dumps(identity))
    return manifest,base,candidate


def test_matched_text_evidence_comparison(tmp_path):
    manifest,base,candidate=text_fixture(tmp_path)
    assert compare(manifest,base,candidate,100)['comparison']['valid/noul.True']['delta']==1


@pytest.mark.parametrize('field,value,match',[
    ('input_mode','asr-text','ASR base'),
    ('enable_thinking',True,'text protocol differs'),
    ('cpu_threads',8,'text protocol differs'),
    ('model_revision','changed','text protocol differs'),
    ('policy_id','other-weights','text model identity differs'),
    ('checkpoint_files',{'weights':'other-hash'},'text model identity differs'),
    ('source_sha256',{'text_baseline.py':'changed'},'text prompt implementation differs')])
def test_text_contrast_rejects_additional_changes(tmp_path,field,value,match):
    manifest,base,candidate=text_fixture(tmp_path)
    path=candidate/'identity.json';identity=json.loads(path.read_text())
    if field in identity['config']:identity['config'][field]=value
    else:identity[field]=value
    path.write_text(json.dumps(identity))
    with pytest.raises(ValueError,match=match):compare(manifest,base,candidate,100)
