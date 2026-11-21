import threading

import pytest

from multimodal_jev.evaluation_campaign import CampaignTimeline,ResidentEvaluationInputs,run_campaign
from tools.reports.context_macro_evaluation import evaluate
from test_context_macro_evaluation import setup


def test_real_macro_transport_reuses_inputs_and_pool_without_journal_collisions(tmp_path,monkeypatch):
    pool,population,encode,args,clients=setup(tmp_path,monkeypatch,n=5)
    loaded=[]
    def counting(asset):loaded.append(asset['id']);return encode(asset)
    timeline=CampaignTimeline()
    with timeline.stage('input_load'):
        resident=ResidentEvaluationInputs(population,counting,io_workers=2,max_bytes=1024**2)
    jobs=[dict(job_id=f'point-{i}',population=population,encode=resident,
               evaluation=dict(args,output=tmp_path/f'eval-{i}')) for i in range(2)]
    audited=[]
    result=run_campaign(pool,jobs,evaluate=evaluate,
                        audit=lambda j,r:audited.append(j['job_id']),timeline=timeline,cpu_workers=2)
    assert sorted(loaded)==[str(i) for i in range(5)]
    assert resident.loads==5 and resident.hits==10
    assert len(audited)==2 and len(result['results'])==2
    assert pool.phase=='ready'  # Owner closes only at campaign end.
    assert all(c.phase!='closed' for c in clients)
    assert resident(population[0])[0]['inputs']['input_ids'].data_ptr()==resident(population[0])[0]['inputs']['input_ids'].data_ptr()
    assert {'input_load','inference','output_write','cpu_audit'}<=result['timing']['stage_work_seconds'].keys()


def test_input_budget_and_identity_fail_closed(tmp_path,monkeypatch):
    _,population,encode,_,_=setup(tmp_path,monkeypatch,n=2)
    with pytest.raises(ValueError,match='budget'):
        ResidentEvaluationInputs(population,encode,io_workers=1,max_bytes=1)
    r=ResidentEvaluationInputs(population,encode,io_workers=1,max_bytes=1024**2)
    with pytest.raises(ValueError,match='identity'):
        r(dict(population[0],audio_sha256='changed'))


def test_policy_switch_and_cpu_overlap_are_real_and_audits_drained():
    started=threading.Event();second=threading.Event();events=[]
    class Pool:
        bundle={'policy_id':'p0'}
        def switch(self,**kw):events.append(('switch',kw['policy_id']));self.bundle={'policy_id':kw['policy_id']}
    def evaluation(pool,*a,**kw):
        events.append(('eval',kw['policy_id']))
        if kw['policy_id']=='p1':
            assert started.wait(2);second.set()
        return kw['policy_id']
    def audit(job,result):
        if result=='p0':started.set();assert second.wait(2)
        return result
    jobs=[dict(job_id=f'job-{i}',snapshot=f's{i}',adapter=f'a{i}',population=[],encode=None,
               evaluation=dict(policy_id=f'p{i}')) for i in range(2)]
    report=run_campaign(Pool(),jobs,evaluate=evaluation,audit=audit,timeline=CampaignTimeline(),cpu_workers=2)
    assert events==[('eval','p0'),('switch','p1'),('eval','p1')]
    assert report['audits']==['p0','p1']
