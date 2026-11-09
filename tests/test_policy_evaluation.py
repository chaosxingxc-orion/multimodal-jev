import json
from pathlib import Path
from multimodal_jev.policy_evaluation import prediction_record
from multimodal_jev.evaluate import evaluate
from multimodal_jev.task_schema import JevExample,DecisionSchema,Candidate


def test_bad_answer_never_becomes_correct_null(tmp_path):
    schema=DecisionSchema('s','choice','Choose',(Candidate('a','A'),Candidate('b','B')))
    example=JevExample('x','valid','fixture.wav',schema,None)
    manifest=tmp_path/'valid.jsonl';manifest.write_text(json.dumps(example.to_record())+'\n')
    predictions=tmp_path/'predictions.jsonl'
    for answer in ['ASR transcription','{"value":null,"value":"a"}','{"value":null,"extra":1}']:
        predictions.write_text(json.dumps(prediction_record('x',answer))+'\n')
        stats=evaluate(manifest,predictions)['by_variant']['all']
        assert stats['correct']==0 and stats['refusals']==0
    predictions.write_text(json.dumps(prediction_record('x','{"value":null}'))+'\n')
    stats=evaluate(manifest,predictions)['by_variant']['all']
    assert stats['correct']==1 and stats['correct_refusals']==1


def test_format_invalid_marker_cannot_claim_valid_value(tmp_path):
    schema=DecisionSchema('s','noul','True?')
    ex=JevExample('x','valid','fixture.wav',schema,True)
    manifest=tmp_path/'valid';manifest.write_text(json.dumps(ex.to_record())+'\n')
    predictions=tmp_path/'predictions';predictions.write_text(json.dumps({'example_id':'x','value':True,'format_valid':False})+'\n')
    assert evaluate(manifest,predictions)['by_variant']['all']['correct']==0


def test_raw_answer_overrides_claimed_value_and_invalid_confidence(tmp_path):
    ex=JevExample('x','valid','fixture.wav',DecisionSchema('s','noul','True?'),True)
    manifest=tmp_path/'valid';manifest.write_text(json.dumps(ex.to_record())+'\n')
    predictions=tmp_path/'predictions'
    for record in [dict(value=True,raw_text='{"value":false}'),
                   dict(value=True,raw_text='not JSON'),
                   dict(value=True,confidence=float('nan'))]:
        predictions.write_text(json.dumps({'example_id':'x',**record})+'\n')
        assert evaluate(manifest,predictions)['by_variant']['all']['correct']==0


def test_binary_pairs_and_null_strata_keep_degenerate_failures(tmp_path):
    examples=[JevExample(str(i),'valid','fixture.wav',DecisionSchema(str(i),'noul',str(i)),target,
                        {'variant':variant,'source_group':'g'})
              for i,(variant,target) in enumerate([('noul.True',True),('noul.False',False),('unknown',None)])]
    manifest=tmp_path/'valid';manifest.write_text(''.join(json.dumps(ex.to_record())+'\n' for ex in examples))
    predictions=tmp_path/'predictions'
    for values,expected in [([True,True,None],0),([True,False,None],1)]:
        predictions.write_text(''.join(json.dumps({'example_id':ex.example_id,'value':value})+'\n'
                                       for ex,value in zip(examples,values)))
        metrics=evaluate(manifest,predictions)
        assert metrics['pairs']['noul']['both_correct_rate']==expected
        assert metrics['by_target_presence']['unknown.target_null']['accuracy']==1
    predictions.write_text(json.dumps({'example_id':'0','raw_text':'not JSON'})+'\n')
    counts=evaluate(manifest,predictions)['by_variant']['all']
    assert counts['malformed']==1 and counts['absent']==2 and counts['format_valid_rate']==0
