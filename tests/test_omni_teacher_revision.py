import importlib.util
import json
import sys
from pathlib import Path

import pytest

from multimodal_jev.baseline import file_hash
from test_teacher_candidates import request_fixture

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
spec=importlib.util.spec_from_file_location('omni_revision',Path(__file__).parents[1]/'tools/jev/prepare_omni_teacher_revision.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_revision_preserves_training_feedback_and_rejects_heldout_review(tmp_path,monkeypatch):
    root=request_fixture(tmp_path)
    path=root/'messages.private.json';messages=json.loads(path.read_text())
    context=json.loads(messages[1]['content'])
    context.update(schema='jev-generative-audio-ace-feedback-v1',weight_updates=0,
                   zeros_semantic_gold_available=False)
    messages[1]['content']=json.dumps(context);path.write_text(json.dumps(messages))
    cfg_path=root/'executor-config.private.json';cfg=json.loads(cfg_path.read_text())
    cfg['messages']['sha256']=file_hash(path);cfg_path.write_text(json.dumps(cfg))
    (root/'prepared.json').write_text(json.dumps(dict(messages_sha256=file_hash(path),API_sent=False)))
    result=json.loads((root/'response/result.json').read_text())
    review=dict(response_sha256=result['response_sha256'],evaluation_eligible=[],
        selection_or_test_items_read_for_review=False,parent_prompt_sha256=context['parent_prompt_sha256'],
        candidates=[{'id':'p1','decision':'rejected'}],diagnosis_corrections=['Preserve null semantics'])
    review_path=root/'rejection-review.json';review_path.write_text(json.dumps(review))
    calls=[]
    monkeypatch.setattr(module,'configure',lambda *args:calls.append(args))
    output=tmp_path/'revision'
    receipt=module.prepare(root,output,tmp_path/'executor.py')
    revised=json.loads(json.loads((output/'messages.private.json').read_text())[1]['content'])
    assert revised['all_group_summary']==context['all_group_summary']
    assert revised['representatives']==context['representatives']
    assert revised['representative_evidence_ids']==['train-evidence-001']
    assert revised['split']=='train' and revised['selection_or_test_feedback'] is False
    assert receipt['previous_usage']==result['usage'] and len(calls)==1
    assert (output/'messages.private.json').stat().st_mode & 0o077==0
    review['selection_or_test_items_read_for_review']=True;review_path.write_text(json.dumps(review))
    with pytest.raises(ValueError,match='TRAIN-only'):
        module.prepare(root,tmp_path/'invalid',tmp_path/'executor.py')
    assert not (tmp_path/'invalid').exists() and len(calls)==1
