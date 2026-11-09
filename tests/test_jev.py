import json
import math
import pytest
from multimodal_jev import Candidate, DecisionSchema, JevExample, score_rollout
from multimodal_jev.reward import compute_score


def example(target='a'):
    return JevExample('x','train','/fixture.wav',DecisionSchema('s','choice','Select',
                      (Candidate('a','activate'), Candidate('b','deactivate'))),target)


def test_target_is_not_in_model_input():
    e=example()
    assert set(e.model_input()) == {'audio','schema'}
    assert JevExample.from_record(e.to_record()) == e


def test_rebinding_changes_correct_target_not_position():
    e=example()
    swapped=JevExample('y','train','/fixture.wav',DecisionSchema('s','choice','Select',
            (Candidate('b','activate'),Candidate('a','deactivate'))),'b')
    assert score_rollout(e,'a')['semantic']==1
    assert score_rollout(swapped,'a')['semantic']==0
    assert score_rollout(swapped,'b')['semantic']==1


def test_refuse_everything_does_not_win():
    e=example()
    assert score_rollout(e,None)['total'] < score_rollout(e,'a')['total']
    assert score_rollout(example(None),None)['total'] > score_rollout(example(None),'a')['total']


def test_schema_types_unknown_and_calibration():
    binary=DecisionSchema('b','noul','True?')
    assert binary.accepts(True) and binary.accepts(None)
    assert not binary.accepts(1) and not binary.accepts('true')
    assert score_rollout(example(),'invalid')['total'] < 0
    assert score_rollout(example(),'b',confidence=1)['total'] < score_rollout(example(),'b',confidence=0)['total']
    for confidence in [math.nan, math.inf,-1,2,True]:
        with pytest.raises(ValueError): score_rollout(example(),'a',confidence=confidence)
    with pytest.raises(ValueError): DecisionSchema('s','score','Rate',levels=(1,1))


def test_verl_reward_rejects_malformed_output_and_claimed_execution():
    gt=json.dumps(example().to_record())
    assert compute_score('fsc','{"value":"a"}',gt)>0
    for solution in ['a','{}','{"value":"a","executed":true}', '{"value":"a","confidence":NaN}']:
        assert compute_score('fsc',solution,gt)==-1


def test_fsc_preserves_splits_and_constructs_pairs(tmp_path):
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('fsc_builder',Path(__file__).parents[1]/'tools/jev/build_fsc_manifest.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    root=tmp_path/'fsc';(root/'data').mkdir(parents=True)
    for split in ('train','valid','test'):
        (root/f'{split}1.wav').write_bytes((split+'1').encode())
        (root/f'{split}2.wav').write_bytes((split+'2').encode())
        (root/'data'/f'{split}_data.csv').write_text('path,speakerId,transcription,action,object,location\n'+
            f'{split}1.wav,{split},turn on,activate,lights,none\n'+
            f'{split}2.wav,{split},turn off,deactivate,lights,none\n')
    # Subset choice needs two remaining candidates, so omit missing variant here.
    config={'protocol':'test','variants':['permutation','rebind','noul']}
    out,receipt=mod.build(root,tmp_path/'out',config)
    assert receipt['splits']['test']['decisions']==10
    records=[JevExample.from_record(json.loads(x)) for x in (out/'train.jsonl').read_text().splitlines()]
    assert records[0].target != records[2].target
    assert all(x.split=='train' for x in records)
    with pytest.raises(FileExistsError): mod.build(root,tmp_path/'out',config)
    (root/'data'/'valid_data.csv').write_text((root/'data'/'train_data.csv').read_text())
    with pytest.raises(ValueError,match='cross-split'): mod.build(root,tmp_path/'out',config)


def test_missing_predictions_are_failures_even_for_unknown(tmp_path):
    from multimodal_jev.evaluate import evaluate
    manifest=tmp_path/'data.jsonl'; predictions=tmp_path/'predictions.jsonl'
    manifest.write_text(json.dumps(example(None).to_record())+'\n')
    predictions.write_text('')
    result=evaluate(manifest,predictions)['by_variant']['all']
    assert result['accuracy']==0 and result['missing']==1 and result['refusals']==0
    predictions.write_text(json.dumps({'example_id':'x','value':None})+'\n')
    result=evaluate(manifest,predictions)['by_variant']['all']
    assert result['accuracy']==1 and result['refusals']==1


def test_strict_json_parser_cannot_reward_duplicate_keys_or_nan():
    from multimodal_jev.answers import parse_answer,AnswerError
    gt=json.dumps(example().to_record())
    for text in ['{"value":"b","value":"a"}', '{"value":"a","confidence":null}',
                 '{"value":NaN}', '{"value":1e999}', '```json\n{"value":"a"}\n```']:
        with pytest.raises(AnswerError):parse_answer(text)
        assert compute_score('fsc',text,gt)==-1
    assert parse_answer('{"value":null}')['value'] is None
