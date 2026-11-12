from types import SimpleNamespace
import pytest
from multimodal_jev.span_training_session import SpanTrainingSession


def fixture():
    actor=SimpleNamespace(config=SimpleNamespace(jev_policy_id='p0'),update_policy=lambda batch: {'updated':True})
    session=SpanTrainingSession(actor,policy_id='p0',completed_updates=0,gather=lambda value:[value]*4)
    return actor,session


def test_update_must_publish_and_pass_gate_before_reuse():
    actor,s=fixture();s.update(None)
    with pytest.raises(ValueError):s.update(None)
    with pytest.raises(ValueError):s.checkpoint_cursor(schedule_sha256='s',checkpoint_id='c')
    s.publish('p1')
    assert actor.config.jev_policy_id=='p0'
    s.accept(lambda policy,step:dict(passed=True,policy_id=policy,updates=step))
    cursor=s.checkpoint_cursor(schedule_sha256='s',checkpoint_id='c')
    restored=SpanTrainingSession.restore(actor,cursor,schedule_sha256='s',checkpoint_id='c',gather=s.gather)
    assert restored.completed_updates==1 and restored.policy_id=='p1'
    with pytest.raises(ValueError):SpanTrainingSession.restore(actor,cursor,schedule_sha256='different',checkpoint_id='c',gather=s.gather)


def test_ambiguous_optimizer_failure_never_retries():
    actor,s=fixture()
    def fail(batch):raise RuntimeError('possibly already stepped')
    actor.update_policy=fail
    with pytest.raises(RuntimeError):s.update(None)
    assert s.phase=='failed'
    with pytest.raises(ValueError):s.update(None)


@pytest.mark.parametrize('bad',['policy','step','failed'])
def test_gate_cannot_advance_wrong_policy(bad):
    actor,s=fixture();s.update(None);s.publish('p1')
    with pytest.raises(ValueError):
        s.accept(lambda p,n:dict(passed=bad!='failed',policy_id='wrong' if bad=='policy' else p,updates=2 if bad=='step' else n))
    assert s.phase=='failed' and actor.config.jev_policy_id=='p0'


def test_rank_state_disagreement_precedes_actor_call():
    actor,s=fixture();actor.update_policy=lambda b:pytest.fail('entered optimizer despite rank mismatch')
    s.gather=lambda value:[value,dict(value,completed_updates=2)]
    with pytest.raises(ValueError,match='collective'):s.update(None)


def test_same_export_identity_preserves_no_change_update():
    actor,s=fixture();s.update(None);s.publish('p0')
    s.accept(lambda p,n:dict(passed=True,policy_id=p,updates=n))
    assert s.completed_updates==1 and s.policy_id=='p0'


def test_different_acceptance_evidence_across_ranks_fails_closed():
    actor,s=fixture();s.update(None);s.publish('p1')
    def gather(value):
        if value['operation'].startswith('accept_result:'):
            return [value,dict(value,operation='accept_result:different-gate-receipt')]
        return [value]*4
    s.gather=gather
    with pytest.raises(ValueError,match='collective'):
        s.accept(lambda p,n:dict(passed=True,policy_id=p,updates=n,files={'gate':'sha256-a'}))
    assert s.phase=='failed' and actor.config.jev_policy_id=='p0'


@pytest.mark.parametrize('extra',[{'metric':float('nan')},{'metric':float('inf')},{'updates':True}])
def test_invalid_receipt_cannot_advance_session(extra):
    actor,s=fixture();s.update(None);s.publish('p1')
    with pytest.raises(ValueError):
        s.accept(lambda p,n:dict(dict(passed=True,policy_id=p,updates=n),**extra))
    assert s.phase=='failed' and actor.config.jev_policy_id=='p0'


def residency_fixture():
    actor,s=fixture();s.update(None);s.publish('p1');events=[]
    def offload():events.append('offload')
    def reload():events.append('reload')
    return actor,s,SimpleNamespace(offload=offload,reload=reload),events


def test_offloaded_verification_precedes_reload_and_policy_acceptance():
    actor,s,r,events=residency_fixture()
    def verify(p,n):
        events.append('verify')
        assert s.phase=='verifying_offloaded' and actor.config.jev_policy_id=='p0'
        with pytest.raises(ValueError):s.update(None)
        return dict(passed=True,policy_id=p,updates=n,gate_hash='fixed')
    receipt=s.accept_offloaded(r,verify)
    assert events==['offload','verify','reload'] and receipt['gate_hash']=='fixed'
    assert s.phase=='accepted' and actor.config.jev_policy_id=='p1'


@pytest.mark.parametrize('stage',['offload','verify','reload'])
def test_offloaded_failure_never_advances_policy_or_retries(stage):
    actor,s,r,events=residency_fixture()
    def failure(*a):events.append('failed');raise ValueError('corruption')
    verify=lambda p,n:dict(passed=True,policy_id=p,updates=n)
    if stage=='verify':verify=failure
    else:setattr(r,stage,failure)
    with pytest.raises(ValueError):s.accept_offloaded(r,verify)
    assert s.phase=='failed' and actor.config.jev_policy_id=='p0'
    prior=list(events)
    with pytest.raises(ValueError):s.accept_offloaded(r,verify)
    assert events==prior


def test_peer_reload_failure_blocks_local_policy_acceptance():
    actor,s,r,events=residency_fixture()
    def gather(value):
        if value['operation']=='offloaded_reload':return [value,dict(value,error='peer failed')]
        return [value]*4
    s.gather=gather
    with pytest.raises(ValueError):s.accept_offloaded(r,lambda p,n:dict(passed=True,policy_id=p,updates=n))
    assert s.phase=='failed' and actor.config.jev_policy_id=='p0'
