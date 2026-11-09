import json
from dataclasses import replace
from pathlib import Path
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.text_baseline import messages,load_evidence
from multimodal_jev.task_schema import JevExample,DecisionSchema


def fixture(tmp_path):
    examples=[JevExample(str(i),'valid','not-read.wav',DecisionSchema('s'+str(i),'noul','Question '+str(i)),
        True,{'audio_sha256':'same-audio','transcript_reference':'GOLD_REFERENCE_SECRET','source_group':'g'}) for i in range(2)]
    manifest=tmp_path/'manifest.jsonl';manifest.write_text(''.join(json.dumps(e.to_record())+'\n' for e in examples))
    run=tmp_path/'asr';run.mkdir();pred=run/'predictions.jsonl'
    rows=[{'example_id':ex.example_id,'audio_sha256':'same-audio','transcription':'actual ASR',
           'reference':'NEVER_USE_ASR_REFERENCE'} for ex in examples]
    pred.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (run/'terminal.json').write_text('{"exit_code":0}')
    (run/'identity.json').write_text(json.dumps({'manifest_sha256':file_hash(manifest)}))
    (run/'result.json').write_text(json.dumps({'status':'passed','records':2,'predictions_sha256':file_hash(pred)}))
    return {'manifest':str(manifest),'input_mode':'asr-text','asr_run':str(run)},examples


def test_asr_prompt_cannot_use_reference_or_target(tmp_path):
    config,examples=fixture(tmp_path);loaded,texts,identity=load_evidence(config)
    prompt=messages(loaded[0],texts['0']);encoded=json.dumps(prompt)
    assert 'actual ASR' in encoded and 'SECRET' not in encoded and 'NEVER_USE' not in encoded
    changed=replace(loaded[0],target=False,metadata={'transcript_reference':'changed', 'hidden_answer':'target'})
    assert messages(changed,texts['0'])==prompt
    assert 'audio' not in prompt[1]['content'] and 'noul' in prompt[0]['content']
    assert identity['mode']=='asr-text'


def test_reference_condition_must_be_explicit(tmp_path):
    config,_=fixture(tmp_path);config['input_mode']='reference-text'
    with pytest.raises(ValueError,match='must not use ASR'):load_evidence(config)
    config['asr_run']=None
    _,texts,identity=load_evidence(config)
    assert set(texts.values())=={'GOLD_REFERENCE_SECRET'} and identity['mode']=='reference-text'


@pytest.mark.parametrize('change,match',[('hash','predictions changed'),('incomplete','complete aligned'),
    ('duplicate','IDs'),('wrong_audio','audio identity'),('inconsistent','inconsistent'),('failed','completed ASR')])
def test_bad_asr_evidence_is_rejected(tmp_path,change,match):
    config,_=fixture(tmp_path);run=Path(config['asr_run']);p=run/'predictions.jsonl'
    rows=[json.loads(x) for x in p.read_text().splitlines()]
    if change=='failed':(run/'terminal.json').write_text('{"exit_code":1}')
    else:
        if change=='incomplete':rows=rows[:1]
        elif change=='duplicate':rows.append(rows[0])
        elif change=='wrong_audio':rows[0]['audio_sha256']='wrong'
        else:rows[0]['transcription']='different'
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        if change!='hash':
            d=json.loads((run/'result.json').read_text());d['predictions_sha256']=file_hash(p)
            (run/'result.json').write_text(json.dumps(d))
    with pytest.raises(ValueError,match=match):load_evidence(config)


def test_manifest_and_reference_missing_are_not_silently_repaired(tmp_path):
    config,_=fixture(tmp_path);p=Path(config['manifest'])
    rows=[json.loads(x) for x in p.read_text().splitlines()]
    rows[0]['metadata'].pop('transcript_reference')
    p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='manifest mismatch'):load_evidence(config)
    config.update(input_mode='reference-text',asr_run=None)
    with pytest.raises(ValueError,match='reference transcript missing'):load_evidence(config)
    rows[0]['split']='test';p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='valid-only'):load_evidence(config)
