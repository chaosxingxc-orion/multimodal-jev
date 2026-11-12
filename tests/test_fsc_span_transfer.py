import copy
import json
from pathlib import Path
import pytest
from multimodal_jev.fsc_span_transfer import compile_request,score_one,validate,FIELDS


def config():return json.loads((Path(__file__).resolve().parents[1]/'configs/tasks/fsc-span-transfer-v1.json').read_text())


def test_templates_share_final_knowledge_and_have_no_gold_input():
    cfg=config();d,db=compile_request('audio.wav',cfg);q,qb=compile_request('audio.wav',cfg,'structured')
    assert d.questions[-1]==q.questions[-1] and db['final']==qb['final']
    assert [len(x.values) for x in q.questions]==[6,14,4,31]
    assert None not in q.questions[1].values and 'none' in q.questions[1].values
    assert 'target' not in json.dumps(q.model_input()) and 'transcription' not in json.dumps(q.model_input())


@pytest.mark.parametrize('variant',['native','permutation','rebind'])
def test_all_31_semantics_survive_output_rebinding(variant):
    cfg=config();request,binding=compile_request('audio.wav',cfg,'structured',variant=variant,seed=317)
    for command in cfg['commands']:
        gold=command['values']
        predicted={f:next(k for k,v in binding[f].items() if v==gold[f]) for f in FIELDS}
        predicted['final']=next(k for k,v in binding['final'].items() if v==gold)
        result=score_one(predicted,binding,gold)
        assert result['final_correct'] and result['compatible'] and all(result['process_correct'].values())
    if variant=='rebind':
        native,_=compile_request('audio.wav',cfg,'structured')
        for a,b in zip(native.questions,request.questions,strict=True):
            assert a.schema.candidates[0].description!=b.schema.candidates[0].description


def test_components_do_not_repair_wrong_final():
    cfg=config();_,binding=compile_request('audio.wav',cfg,'structured');gold=cfg['commands'][0]['values']
    pred={f:next(k for k,v in binding[f].items() if v==gold[f]) for f in FIELDS}
    pred['final']=cfg['commands'][1]['id'];result=score_one(pred,binding,gold)
    assert all(result['process_correct'].values()) and not result['final_correct'] and not result['compatible']
    with pytest.raises(ValueError):score_one({**pred,'final':'SLURP_ID'},binding,gold)
    damaged=copy.deepcopy(cfg);damaged['commands'].pop()
    with pytest.raises(ValueError,match='31-command'):validate(damaged)
