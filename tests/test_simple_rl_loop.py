"""CPU numerical continuity and failure controls, not native verl/NPU acceptance."""
from types import SimpleNamespace
import copy
import pytest
torch = pytest.importorskip('torch')
from multimodal_jev.simple_rl_loop import run
from multimodal_jev.span_training_session import SpanTrainingSession


class Schedule:
    total_updates = 4
    sha256 = 'fixed-schedule'
    def at(self, step):
        return [step], dict(update=step, next_optimizer_update=step+1,
            total_updates=4, schedule_sha256=self.sha256, batch_index=step,
            coefficient=1.-.1*step, real_requests=1)


def fixture():
    parameter = torch.nn.Parameter(torch.tensor([.3, -.2]))
    optimizer = torch.optim.AdamW([parameter], lr=.01, foreach=False, fused=False)
    actor = SimpleNamespace(config=SimpleNamespace(jev_policy_id='policy0'))
    def update(batch):
        optimizer.zero_grad()
        loss = -parameter.log_softmax(0)[batch['target']]
        loss.backward();optimizer.step()
        return dict(loss=float(loss.detach()))
    actor.update_policy = update
    session = SpanTrainingSession(actor, policy_id='policy0', completed_updates=0,
                                  gather=lambda value: [value]*4)
    exports = {}; records = []; states = []
    def collect(slots, cursor, policy):
        assert slots == [cursor['update']] and policy == actor.config.jev_policy_id
        assert cursor['coefficient'] == 1.-.1*cursor['update']
        return dict(target=cursor['update']%2)
    def export(session, batch, metrics):
        policy = 'policy'+str(session.completed_updates)
        exports[policy] = parameter.detach().clone()
        return policy
    def verify(policy, updates, batch):
        assert torch.equal(exports[policy], parameter.detach())
        assert optimizer.state[parameter]['step'].item() == updates
        return dict(passed=True, policy_id=policy, updates=updates)
    def checkpoint(session, cursor):
        state = dict(parameter=parameter.detach().clone(), optimizer=copy.deepcopy(optimizer.state_dict()),
                     cursor=session.checkpoint_cursor(schedule_sha256='fixed-schedule', checkpoint_id='state'))
        states.append(state)
        return dict(completed_updates=session.completed_updates)
    callbacks = dict(collect=collect, export=export, verify=verify, checkpoint=checkpoint,
                     checkpoint_interval=2, record=records.append)
    return parameter, optimizer, session, callbacks, records, states


def test_continuous_optimizer_and_exact_resume_match_uninterrupted():
    p, opt, session, callbacks, records, states = fixture()
    run(session, Schedule(), stop_update=4, **callbacks)
    assert [r['completed_updates'] for r in records] == [1,2,3,4]
    assert [r['checkpoint'] is not None for r in records] == [False,True,False,True]
    assert opt.state[p]['step'].item() == 4
    q, resumed_opt, resumed, resumed_callbacks, _, saved = fixture()
    run(resumed, Schedule(), stop_update=2, **resumed_callbacks)
    # Actual tensor/Adam reload before continuing the same schedule.
    with torch.no_grad(): q.copy_(saved[-1]['parameter'])
    resumed_opt.load_state_dict(saved[-1]['optimizer'])
    resumed = SpanTrainingSession.restore(resumed.actor, saved[-1]['cursor'],
        schedule_sha256='fixed-schedule', checkpoint_id='state', gather=resumed.gather)
    run(resumed, Schedule(), stop_update=4, **resumed_callbacks)
    assert torch.equal(p,q)
    for key in ('step','exp_avg','exp_avg_sq'):
        assert torch.equal(opt.state[p][key], resumed_opt.state[q][key])


@pytest.mark.parametrize('stage',['collect','export','verify','checkpoint','record','evaluate'])
def test_failure_poisoning_prevents_any_implicit_second_update(stage):
    p, opt, session, callbacks, records, _ = fixture()
    callbacks['checkpoint_interval'] = 1
    def fail(*args): raise RuntimeError('injected '+stage)
    callbacks[stage] = fail
    with pytest.raises(ValueError,match='injected'): run(session, Schedule(), stop_update=4, **callbacks)
    assert session.phase == 'failed'
    actual = opt.state[p]['step'].item() if p in opt.state else 0
    assert actual == (0 if stage == 'collect' else 1)
    with pytest.raises(ValueError): run(session, Schedule(), stop_update=4, **callbacks)
    assert (opt.state[p]['step'].item() if p in opt.state else 0) == actual


def test_peer_collect_failure_rejects_before_optimizer():
    p, opt, session, callbacks, _, _ = fixture()
    def gather(value):
        if value['operation'] == 'simple_loop_collect': return [value,dict(value,error='peer collection failure')]
        return [value]*4
    session.gather = gather
    with pytest.raises(ValueError,match='peer collection'):run(session,Schedule(),stop_update=4,**callbacks)
    assert session.phase == 'failed' and not opt.state


def test_peer_schedule_disagreement_rejected_before_collection():
    p, opt, session, callbacks, _, _ = fixture()
    calls = []
    callbacks['collect'] = lambda *args: calls.append('collect')
    def gather(value):
        if value['operation'].startswith('simple_loop_cursor_agreement:'):
            return [value, dict(value, operation='simple_loop_cursor_agreement:other')]
        return [value]*4
    session.gather = gather
    with pytest.raises(ValueError): run(session,Schedule(),stop_update=4,**callbacks)
    assert not calls and not opt.state and session.phase == 'failed'


def test_epoch_end_checkpoint_is_not_skipped_by_interval():
    _, _, session, callbacks, records, _ = fixture()
    class Boundary(Schedule):
        def at(self, step):
            slots, cursor = super().at(step)
            if step == 0: cursor['batch_index'] = 316
            return slots, cursor
    callbacks['checkpoint_interval'] = 99
    run(session,Boundary(),stop_update=2,**callbacks)
    assert all(r['checkpoint'] is not None for r in records)


def test_invalid_cursor_fails_before_collection_and_optimizer():
    _, opt, session, callbacks, _, _ = fixture()
    class Invalid(Schedule):
        def at(self, step):
            slots, cursor = super().at(step)
            cursor['coefficient'] = float('nan')
            return slots, cursor
    with pytest.raises(ValueError): run(session,Invalid(),stop_update=4,**callbacks)
    assert session.phase == 'failed' and not opt.state


def test_evaluation_runs_after_committed_accepted_state_without_changing_updates():
    parameter, optimizer, session, callbacks, records, states = fixture()
    callbacks['checkpoint_interval'] = 1
    observed = []
    def evaluate(current, report):
        assert current.phase == 'accepted'
        assert report is records[-1] and report['checkpoint'] is not None
        assert torch.equal(states[-1]['parameter'], parameter.detach())
        assert optimizer.state[parameter]['step'].item() == current.completed_updates
        observed.append((current.completed_updates, current.policy_id))
    run(session, Schedule(), stop_update=4, evaluate=evaluate, **callbacks)
    assert observed == [(i, 'policy'+str(i)) for i in range(1, 5)]
    assert optimizer.state[parameter]['step'].item() == 4
