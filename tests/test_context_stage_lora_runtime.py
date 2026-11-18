import copy

import pytest
import torch

from multimodal_jev.context_policy_snapshot import lora_contract
from multimodal_jev.context_stage_lora import from_policy
from tools.reports.context_stage_lora_runtime import configure_stage2


def model():
    from peft import LoraConfig,get_peft_model
    from transformers import Qwen3Config,Qwen3ForCausalLM
    torch.manual_seed(17)
    base=Qwen3ForCausalLM(Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8)).bfloat16().eval()
    # Match the audio model's language_model nesting and ordinary PEFT path.
    from transformers import PreTrainedModel
    class Wrapper(PreTrainedModel):
        config_class=Qwen3Config
        _supports_sdpa=True
        def __init__(self):
            super().__init__(base.config)
            self.model=torch.nn.Module();self.model.language_model=base.model
            self.lm_head=base.lm_head
        def get_input_embeddings(self):return self.model.language_model.embed_tokens
        def prepare_inputs_for_generation(self,*args,**kwargs):return {}
        def forward(self,input_ids,**kwargs):
            out=self.model.language_model(input_ids=input_ids,use_cache=False)
            return self.lm_head(out.last_hidden_state)
    wrapped=Wrapper().bfloat16().eval()
    adapted=get_peft_model(wrapped,LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,bias='none',
        task_type='CAUSAL_LM',target_modules=['model.language_model.layers.0.self_attn.q_proj',
                                           'model.language_model.layers.0.self_attn.v_proj']),autocast_adapter_dtype=False)
    with torch.no_grad():
        for name,p in adapted.named_parameters():
            if '.lora_B.' in name:p.normal_(0,.05)
    contract={'lora':lora_contract(adapted.get_base_model().model)}
    return adapted,contract,from_policy(contract)


@pytest.mark.parametrize('different,reuse',[(False,True),(False,False),(True,False)])
def test_real_peft_transition_preserves_initial_function_and_gradient_scope(different,reuse):
    adapted,contract,first=model();second=dict(first,rank=3,alpha=6) if different else copy.deepcopy(first)
    ids=torch.tensor([[1,2,3,4]])
    before={n:p.detach().clone() for n,p in adapted.named_parameters()}
    with torch.no_grad():expected=adapted(input_ids=ids).clone()
    receipt=configure_stage2(adapted,contract,first,second,reuse_stage1_adapter=reuse)
    assert receipt['plan']['mode']==('reuse' if reuse else 'frozen_stage1_plus_stage2')
    assert torch.equal(adapted(input_ids=ids),expected)
    assert all(torch.equal(dict(adapted.named_parameters())[n],p) for n,p in before.items())
    trainable=[n for n,p in adapted.named_parameters() if p.requires_grad]
    suffix='default' if reuse else 'stage2'
    assert trainable and all('.lora_A.'+suffix+'.' in n or '.lora_B.'+suffix+'.' in n for n in trainable)
    opt=torch.optim.SGD([p for p in adapted.parameters() if p.requires_grad],lr=.1,foreach=False,fused=False)
    adapted(input_ids=ids).float().square().mean().backward();opt.step()
    if not reuse:
        assert all(torch.equal(dict(adapted.named_parameters())[n],p) for n,p in before.items())
        assert any(torch.count_nonzero(p.grad).item() for n,p in adapted.named_parameters()
                   if '.lora_B.stage2.' in n and p.grad is not None)
    else:
        assert set(adapted.peft_config)=={'default'}
        assert any(not torch.equal(dict(adapted.named_parameters())[n],p) for n,p in before.items())
