from copy import deepcopy
from types import SimpleNamespace
import pytest

torch=pytest.importorskip('torch')
from multimodal_jev.omni_actor_forward import compute_log_probs


class Batch:
    def __init__(self,tensors,non_tensors,meta):
        self.batch=tensors;self.non_tensor_batch=non_tensors;self.meta_info=meta
    def select(self,batch_keys,non_tensor_batch_keys):
        return Batch({k:self.batch[k] for k in batch_keys},
                     {k:self.non_tensor_batch[k] for k in non_tensor_batch_keys},self.meta_info)


def fixture():
    responses=torch.tensor([[4,5],[6,0]])
    mask=torch.tensor([[1,1],[1,0]])
    data=Batch(dict(responses=responses,response_mask=mask,old_log_probs=torch.tensor([[-1.,-2.],[-3.,0.]]),
        input_ids=torch.tensor([[1,2,4,5],[0,3,6,0]]),attention_mask=torch.tensor([[1,1,1,1],[0,1,1,0]]),
        position_ids=torch.tensor([[0,1,2,3],[1,0,1,1]])),{'multi_modal_inputs':['two-audio','one-audio']},
        dict(audio_contract='sfc-jev-qwen3-omni-padded-v1',requires_remove_padding=False,requires_sequence_parallel_size=1))
    actor=SimpleNamespace(use_remove_padding=False,use_fused_kernels=False,use_prefix_grouper=False,
                          use_dynamic_bsz=False,ulysses_sequence_parallel_size=1)
    def compute(inputs,calculate_entropy):
        assert not calculate_entropy and 'old_log_probs' not in inputs.batch
        assert inputs.meta_info['temperature']==1. and inputs.meta_info['micro_batch_size']==1
        assert inputs.non_tensor_batch['multi_modal_inputs']==['two-audio','one-audio']
        return {'log_probs':torch.tensor([[-.5,-.7],[-.9,float('nan')]])}
    actor.compute_log_prob=compute
    return actor,data


def test_recomputed_values_do_not_replace_old_logprobs_or_metadata():
    actor,data=fixture();old=data.batch['old_log_probs'].clone();meta=deepcopy(data.meta_info)
    result=compute_log_probs(actor,data,micro_batch_size=1,pad_token_id=0)
    torch.testing.assert_close(result,torch.tensor([[-.5,-.7],[-.9,0.]]))
    assert torch.equal(data.batch['old_log_probs'],old) and data.meta_info==meta


@pytest.mark.parametrize('kind',['rmpad','fused','prefix','dynamic','sp','contract','mask','tokens','old','output','size'])
def test_incompatible_actor_or_invalid_token_contract_rejected(kind):
    actor,data=fixture()
    names=dict(rmpad='use_remove_padding',fused='use_fused_kernels',prefix='use_prefix_grouper',dynamic='use_dynamic_bsz')
    if kind in names:setattr(actor,names[kind],True)
    if kind=='sp':actor.ulysses_sequence_parallel_size=2
    if kind=='contract':data.meta_info['audio_contract']='asr'
    if kind=='mask':data.batch['response_mask'][0]=torch.tensor([0,1])
    if kind=='tokens':data.batch['responses'][0,0]=9
    if kind=='old':data.batch['old_log_probs'][0,0]=float('nan')
    if kind=='output':actor.compute_log_prob=lambda *a,**k:{'log_probs':torch.full((2,2),float('nan'))}
    with pytest.raises(ValueError):compute_log_probs(actor,data,micro_batch_size=0 if kind=='size' else 1,pad_token_id=0)
