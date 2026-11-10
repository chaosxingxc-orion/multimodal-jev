import json
from pathlib import Path
import sys

import pytest

from multimodal_jev.baseline import file_hash

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
import prepare_omni_teacher_budget_revision as module


def test_budget_revision_preserves_identical_messages_without_replay(tmp_path,monkeypatch):
    source=tmp_path/'prior';source.mkdir();(source/'response').mkdir()
    messages=source/'messages.private.json';messages.write_text('[{"role":"system","content":"frozen TRAIN"}]')
    wire=source/'response/response.sse';wire.write_text('truncated fixture')
    result=dict(state='response_received',http_status=200,finish_reasons=['length'],
        response_sha256=file_hash(wire),usage={'total_tokens':74397})
    (source/'response/result.json').write_text(json.dumps(result))
    prepared=dict(messages_sha256=file_hash(messages),selection_or_test_feedback_used=False)
    (source/'prepared.json').write_text(json.dumps(prepared))
    cfg=dict(thinking={'type':'enabled'},max_tokens=32768,
             messages={'path':str(messages),'sha256':file_hash(messages)})
    (source/'executor-config.private.json').write_text(json.dumps(cfg))
    calls=[];monkeypatch.setattr(module,'configure',lambda *args,**kwargs:calls.append(kwargs))
    output=tmp_path/'revision';receipt=module.prepare(source,output,tmp_path/'executor')
    assert (output/'messages.private.json').read_bytes()==messages.read_bytes()
    assert receipt['previous_usage']==result['usage']
    assert calls==[dict(thinking='disabled',max_tokens=8192)]
    assert (output/'messages.private.json').stat().st_mode & 0o077==0
    with pytest.raises(FileExistsError):module.prepare(source,output,tmp_path/'executor')
    result['finish_reasons']=['stop'];(source/'response/result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError,match='truncated'):module.prepare(source,tmp_path/'invalid',tmp_path/'executor')
    assert not (tmp_path/'invalid').exists()
