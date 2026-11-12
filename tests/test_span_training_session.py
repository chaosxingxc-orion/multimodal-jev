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
