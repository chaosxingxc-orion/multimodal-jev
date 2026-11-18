import copy
import json

import pytest
import torch

from multimodal_jev.context_decision_head import ContextDecisionHead
from multimodal_jev.context_decision_model import ContextDecisionModel
import multimodal_jev.context_policy_snapshot as snapshot
from multimodal_jev.context_policy_artifacts import read_execution_bundle, native_peft_payload


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(20,8).requires_grad_(False)
        self.language_model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        self.language_model.layers = torch.nn.ModuleList([layer])
        for name in ('q_proj','v_proj'):
            m = torch.nn.Module()
            m.lora_A = torch.nn.ModuleDict({'default':torch.nn.Linear(8,2,bias=False)})
            m.lora_B = torch.nn.ModuleDict({'default':torch.nn.Linear(2,8,bias=False)})
            m.r = {'default':2}; m.lora_alpha = {'default':4}; m.scaling = {'default':2.}
            m.active_adapters = ['default']; m.disable_adapters = m.merged = m.fan_in_fan_out = False
            m.use_dora = {'default':False}; m.lora_dropout = torch.nn.ModuleDict({'default':torch.nn.Identity()})
            setattr(layer.self_attn,name,m)

    def get_input_embeddings(self): return self.embedding


def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(snapshot,'check_paths',lambda paths:[])
    model = ContextDecisionModel(Backbone(),ContextDecisionHead(8,2,16),19).bfloat16().eval()
    p = tmp_path/'processor'; p.mkdir()
    (p/'tokenizer.json').write_text(json.dumps(dict(added_tokens=[dict(id=19,content='<|jev_decision|>',special=True)])))
    (p/'tokenizer_config.json').write_text('{}')
    contract = dict(base_id='a'*64,runtime_profile='test-bf16',request_version='test-v1',
        branch=dict(hidden_size=8,num_heads=2,ffn_size=16,residual_scale=.1,layers=1,
            position_encoding='inherited_backbone_states_no_extra_branch_encoding',dropout=0.,
            readout='decision_marker_itself',attention_scope='same_request_valid_causal_prefix_including_marker'),
        token=dict(id=19,text='<|jev_decision|>',row_trainable=False,registration_status='test_only'),
        output=dict(field='prob',activation='sigmoid',normalize_across_candidates=False),
        lora=snapshot.lora_contract(model.backbone))
    path = tmp_path/'policy'; identity = snapshot.export_policy(path,model,p,contract)
    return model,path,identity,contract


def test_authentication_and_native_unmerged_lora_algebra(tmp_path,monkeypatch):
    model,path,identity,contract = fixture(tmp_path,monkeypatch)
    rng = torch.random.get_rng_state().clone()
    bundle = read_execution_bundle(path,expected_policy_id=identity,expected_contract=contract,expected_base_id='a'*64)
    assert torch.equal(rng,torch.random.get_rng_state())
    config,weights = native_peft_payload(bundle)
    assert config['r'] == 2 and config['lora_alpha'] == 4
    x = torch.randn(7,8,dtype=torch.float64)
    for name,module in model.backbone.named_modules():
        if not hasattr(module,'lora_A'): continue
        a = weights[f'base_model.model.model.{name}.lora_A.weight']
        b = weights[f'base_model.model.model.{name}.lora_B.weight']
        assert torch.equal(a,module.lora_A['default'].weight)
        assert torch.equal(b,module.lora_B['default'].weight)
        actual = ((x @ a.double().T) @ b.double().T)*(config['lora_alpha']/config['r'])
        expected = ((x @ module.lora_A['default'].weight.double().T) @
                    module.lora_B['default'].weight.double().T)*module.scaling['default']
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)


@pytest.mark.parametrize('fault',['policy','base','processor','extra','weight','contract','symlink'])
def test_integrity_failure_rejected(tmp_path,monkeypatch,fault):
    _,path,identity,contract = fixture(tmp_path,monkeypatch)
    base = 'a'*64
    if fault == 'policy': identity = 'b'*64
    if fault == 'base': base = 'b'*64
    if fault == 'processor': (path/'processor/tokenizer_config.json').write_text('{"changed":true}')
    if fault == 'extra': (path/'extra').write_text('extra')
    if fault == 'weight':
        with (path/'execution.pt').open('ab') as f: f.write(b'changed')
    if fault == 'contract':
        contract = copy.deepcopy(contract); contract['branch']['residual_scale'] = .2
    if fault == 'symlink': (path/'link').symlink_to(path/'execution.pt')
    with pytest.raises(ValueError):
        read_execution_bundle(path,expected_policy_id=identity,expected_contract=contract,expected_base_id=base)


def test_authenticated_unsupported_scaling_refused(tmp_path,monkeypatch):
    model,path,_,contract = fixture(tmp_path,monkeypatch)
    model.backbone.language_model.layers[0].self_attn.q_proj.scaling['default'] = 3.
    contract['lora'] = snapshot.lora_contract(model.backbone)
    new_path = tmp_path/'scaled'
    identity = snapshot.export_policy(new_path,model,path/'processor',contract)
    with pytest.raises(ValueError,match='rank/alpha'):
        read_execution_bundle(new_path,expected_policy_id=identity,expected_contract=contract,expected_base_id='a'*64)
