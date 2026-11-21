import hashlib
import json

import pytest

from tools.reports.audit_resident_evaluation_candidate import audit


def test_complete_campaign_independent_audit_refuses_missing_or_changed_outputs(tmp_path,monkeypatch):
    from multimodal_jev import local_checkpoint
    monkeypatch.setattr(local_checkpoint,'check_paths',lambda _:None)
    def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
    root=tmp_path/'run';old=tmp_path/'old.json'
    record=dict(id='sample',policy_id='p',request_identity={'audio':'a'},prob=[.2,.8],logits=[-1.,1.])
    write(old,dict(records=[record]));sha=hashlib.sha256(old.read_bytes()).hexdigest()
    state=dict(Status='exited',ExitCode=0,OOMKilled=False,StartedAt='2026-01-01T00:00:00Z',FinishedAt='2026-01-01T00:00:10Z')
    oldinspect=tmp_path/'oldinspect.json';write(oldinspect,{'State':state})
    write(root/'experiment.json',dict(points=[dict(job_id='point',policy_id='p',predictions={'sha256':sha})],limit=None,expected_population=1,devices=[4]))
    write(root/'terminal.json',dict(exit_code=0,wall_seconds=11,stages={'collection':1}))
    write(root/'inspect.json',{'State':state});out=root/'collected'
    write(out/'terminal.json',{'exit_code':0});worker=out/'worker-0'
    write(worker/'terminal.json',{'exit_code':0});write(worker/'result.json',dict(model_initializations=1,input_loads=1,timing={'stage_work_seconds':{'inference':4}}))
    prediction=worker/'point/predictions.json';write(prediction,dict(records=[record]))
    c=dict(run=str(root),prior={'point':dict(predictions=str(old),inspect=str(oldinspect))},output=str(tmp_path/'audit.json'))
    proof=audit(c)
    assert proof['container_speedup']==1 and proof['end_to_end_speedup'] is None and not proof['performance_10x_passed']
    write(prediction,dict(records=[dict(record,logits=[-1.,2.])]))
    with pytest.raises(ValueError,match='outputs differ'):audit(c)
