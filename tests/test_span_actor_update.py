import pytest
torch=pytest.importorskip('torch')
from multimodal_jev.span_actor_update import record_rollout,admit_actor,update_span_actor


def fixture():
    probabilities=(torch.tensor([[.4,.6]]).log(),torch.tensor([[.8,.2]]).log())
    return record_rollout('base:adapter-v1',('exact-input-v1',),probabilities,8,
                          torch.Generator().manual_seed(17))


def test_rollout_owns_probabilities_and_detects_tamper_and_stale_version():
    r=fixture()
    assert all(x['passed'] for x in admit_actor(r,r.candidate_logprobs,r.policy_id,r.request_ids))
    with pytest.raises(ValueError,match='stale'):
        admit_actor(r,r.candidate_logprobs,'adapter-v2',r.request_ids)
    r.old_joint_logprobs[0,0]+=.1
    with pytest.raises(ValueError,match='changed'):
        admit_actor(r,r.candidate_logprobs,r.policy_id,r.request_ids)


def test_failed_parity_prevents_all_optimizer_mutations():
    r=fixture();p=torch.nn.Parameter(torch.ones(1,dtype=torch.bfloat16))
    class ForbiddenOptimizer:
        def zero_grad(self,**kwargs):raise AssertionError('optimizer touched before admission')
        def step(self):raise AssertionError('optimizer touched before admission')
    with pytest.raises(ValueError,match='parity gate'):
        update_span_actor(r,lambda:(torch.tensor([[.5,.5]]).log(),r.candidate_logprobs[1]),
                          ForbiddenOptimizer(),[p],torch.zeros(1,8),r.policy_id,r.request_ids)
    assert p.grad is None and p.item()==1


def test_candidate_drift_checks_unsampled_actions_too():
    r=fixture()
    changed=(torch.tensor([[.41,.59]]).log(),r.candidate_logprobs[1])
    with pytest.raises(ValueError,match='parity gate'):
        admit_actor(r,changed,r.policy_id,r.request_ids)
