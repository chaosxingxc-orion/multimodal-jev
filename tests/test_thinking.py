import hashlib
import json
from dataclasses import replace

import pytest

from multimodal_jev.audio_policy import encode_prompt, hf_prompt, system_text, vllm_request
from multimodal_jev.consumption import canonical, inference_messages, request_from_example, validate_request
from multimodal_jev.direct_readout import prompt
from multimodal_jev.frozen_contract import FrozenContract
from multimodal_jev.task_schema import DecisionSchema, JevExample
from multimodal_jev.thinking import ThinkingSpec


def spec():
    return ThinkingSpec('evidence-v1', ('Check negation scope using the complete audio.',))


def example(thinking=None):
    return JevExample('id', 'train', '/audio.wav', DecisionSchema('type', 'noul', 'Is a person mentioned?'),
                      True, {'source_group':'group', 'audio_sha256':'a'*64,
                             'transcript_reference':'SECRET'}, thinking)


class Processor:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content'][0]['text']+'<|audio_pad|>'

    def __call__(self, **kwargs):
        return kwargs


def test_serialization_and_legacy_prompt_unchanged():
    old=example(); configured=example(spec())
    assert 'thinking' not in old.to_record() and 'thinking' not in old.model_input()
    assert JevExample.from_record(json.loads(json.dumps(old.to_record())))==old
    assert JevExample.from_record(json.loads(json.dumps(configured.to_record())))==configured
    from multimodal_jev.audio_policy import INSTRUCTION
    from multimodal_jev.task_schema import render_schema
    assert system_text(old)=='English\n'+INSTRUCTION+render_schema(old.schema)


def test_full_requirement_changes_prompt_and_request_not_gold():
    ex=example(spec()); audio=object(); processor=Processor()
    req=validate_request(request_from_example(ex))
    hf=hf_prompt(processor,ex,audio)
    rollout=vllm_request(processor,ex,audio)
    assert hf['text']==[rollout['prompt']]
    assert 'Check negation scope' in rollout['prompt']
    assert 'SECRET' not in rollout['prompt']
    assert 'decision_only' in rollout['prompt']
    assert rollout['multi_modal_data']['audio']==(audio,16000)
    other=replace(ex,target=False,metadata={**ex.metadata,'transcript_reference':'OTHER'})
    assert request_from_example(other)==req
    assert system_text(other)==system_text(ex)
    changed=replace(ex,thinking=replace(spec(),requirements=('Check modifier attachment.',)))
    assert system_text(changed)!=system_text(ex)
    assert hashlib.sha256(canonical(request_from_example(changed))).hexdigest()!=hashlib.sha256(canonical(req)).hexdigest()
    # Same identity check used by the verl adapter must distinguish this config.
    assert json.dumps(ex.model_input(),sort_keys=True)!=json.dumps(changed.model_input(),sort_keys=True)
    assert 'Check negation scope' in prompt(req,'Decide')[0]
    assert 'Check negation scope' in FrozenContract({'task_format':'jev'}).audio_instruction(req,'Decide')
    with pytest.raises(ValueError,match='audio-direct'):
        inference_messages(req,'transcript','Decide')


def test_unimplemented_review_fails_all_direct_renderers():
    ex=example(replace(spec(),mode='bounded_review',max_rounds=3))
    req=validate_request(request_from_example(ex))
    for render in (lambda:encode_prompt(Processor(),ex,object()),
                   lambda:prompt(req,'Decide'),
                   lambda:FrozenContract({'task_format':'jev'}).audio_instruction(req,'Decide')):
        with pytest.raises(ValueError,match='controller'):render()


@pytest.mark.parametrize('change', [
    {'version':''}, {'requirements':[]}, {'requirements':'check'},
    {'requirements':['same','same']}, {'requirements':['']},
    {'max_rounds':True}, {'max_rounds':0}, {'max_rounds':2},
    {'mode':'text_cot'}, {'output':'rationale'}, {'gold':True},
])
def test_invalid_or_unsupported_configuration_rejected(change):
    value={**spec().to_dict(),**change}
    with pytest.raises((ValueError,TypeError)):
        ThinkingSpec.from_dict(value)


def test_no_target_fields_inside_thinking():
    req=request_from_example(example(spec()))
    req['input']['thinking']['target']=True
    with pytest.raises(ValueError,match='fields'):validate_request(req)


def test_sft_explicit_schedule_preserves_configuration(tmp_path):
    from multimodal_jev.sft import training_batches
    ex=example(spec())
    path=tmp_path/'train.jsonl'
    path.write_text(json.dumps(ex.to_record())+'\n')
    batches=training_batches({'train_manifest':str(path),'batches':[['id']]})
    assert batches[0][0].thinking==spec()
    assert 'Check negation scope' in system_text(batches[0][0])


def test_export_binds_configuration_and_separates_targets(tmp_path):
    from multimodal_jev.consumption import export_release
    manifests={}
    for split in ('train','valid','test'):
        ex=replace(example(spec()),example_id=split,split=split,
                   metadata={'source_group':split,'audio_sha256':split})
        path=tmp_path/(split+'.jsonl')
        path.write_text(json.dumps(ex.to_record())+'\n'); manifests[split]=path
    out,_=export_release(manifests,tmp_path/'output','thinking-fixture')
    req=json.loads((out/'train.requests.jsonl').read_text())
    target=json.loads((out/'train.targets.jsonl').read_text())
    assert req['input']['thinking']==spec().to_dict()
    assert target['request_sha256']==hashlib.sha256(canonical(req)).hexdigest()
    assert 'target' not in req and 'thinking_sha256' in json.loads((out/'MANIFEST.json').read_text())['identity']
