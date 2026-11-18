import pytest

from multimodal_jev.span_batch_training_config import PROFILE
from multimodal_jev.span_batched_instruction_audit import completed_range, input_policy_id, verify_rank_report


def config():
    return dict(updates=2, global_batch=128, microbatch_per_rank=8, candidate_profile=PROFILE,
                batch_order='length_grouped', checkpoint_interval=2, base_id='base')


def test_new_prefix_cursor_and_nonexport_policy():
    c=config()
    assert completed_range(c,[317,317]) == (0,2,(0,2))
    assert input_policy_id(c,'/new',1,[317,317],lambda p:p.name) == 'base:adapter-step0000:instruction-update1'


@pytest.mark.parametrize('change',[dict(microbatch_per_rank=32),dict(candidate_profile='legacy'),dict(updates=3)])
def test_unadmitted_layout_and_range_rejected(change):
    with pytest.raises(ValueError): completed_range(dict(config(),**change),[317,317])


def test_every_rank_update_and_bf16_hook_required():
    report=dict(passed=True,rank=2,updates=2,executed_updates=2,base_id='base',final_adapter_id='final',
                step_metrics=[dict(update=i,seconds=1.,metrics={'actor/grad_norm':[.2]}) for i in (1,2)],
                execution_dtypes={str(i):['torch.bfloat16']*2 for i in range(112)})
    assert verify_rank_report(report,{'exit_code':0},config(),rank=2,epoch_updates=[317,317],adapter_id='final')==2
    report['step_metrics'].pop()
    with pytest.raises(ValueError):
        verify_rank_report(report,{'exit_code':0},config(),rank=2,epoch_updates=[317,317],adapter_id='final')


@pytest.mark.parametrize('fault',[None,'duplicate','probabilities','summary','missing_micro'])
def test_recovery_raw_probabilities_and_exact_population(fault):
    import math
    from multimodal_jev.span_parity import compare_span_policies
    from multimodal_jev.span_batched_instruction_audit import verify_restored_report
    refs=[dict(id=str(i),candidate_logprobs=[[-math.log(2)]*2]) for i in range(256)]
    records=[dict(id=r['id'],microbatch=m,candidate_logprobs=r['candidate_logprobs'],
                  parity=compare_span_policies(r['candidate_logprobs'],r['candidate_logprobs']))
             for m in (1,8) for r in refs[2::4]]
    report=dict(rank=2,requests=64,comparisons=128,passed=True,records=records,maximum_logprob_error=0.)
    if fault=='duplicate':records[1]=records[0]
    if fault=='probabilities':records[0]['candidate_logprobs']=[[math.log(.9),math.log(.1)]]
    if fault=='summary':report['maximum_logprob_error']=1.
    if fault=='missing_micro':records.pop()
    if fault:
        with pytest.raises(ValueError):verify_restored_report(report,refs,rank=2,microbatch=8)
    else:
        assert verify_restored_report(report,refs,rank=2,microbatch=8)['comparisons']==128
