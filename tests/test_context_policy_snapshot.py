import copy
import json
from types import SimpleNamespace

import pytest
import torch

import multimodal_jev.context_policy_snapshot as snapshot
from multimodal_jev.context_decision_head import ContextDecisionHead
from multimodal_jev.context_decision_model import ContextDecisionModel


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(20,8)
        self.embedding.requires_grad_(False)
        self.q_proj = torch.nn.Module()
        self.q_proj.lora_A = torch.nn.ModuleDict({'default':torch.nn.Linear(8,2,bias=False)})
        self.q_proj.lora_B = torch.nn.ModuleDict({'default':torch.nn.Linear(2,8,bias=False)})
        self.q_proj.r = {'default':2}
        self.q_proj.lora_alpha = {'default':2}
        self.q_proj.scaling = {'default':1.}
        self.q_proj.active_adapters = ['default']
        self.q_proj.disable_adapters = self.q_proj.merged = self.q_proj.fan_in_fan_out = False
        self.q_proj.use_dora = {'default':False}
        self.q_proj.lora_dropout = torch.nn.ModuleDict({'default':torch.nn.Identity()})
    def get_input_embeddings(self):return self.embedding
    def forward(self,input_ids,attention_mask,use_cache,return_dict):
        h=self.embedding(input_ids)
        h=h+self.q_proj.scaling['default']*self.q_proj.lora_B['default'](self.q_proj.lora_A['default'](h))
        return SimpleNamespace(last_hidden_state=h.cumsum(1))


def setup(tmp_path,monkeypatch):
    # Only storage admission is replaced for portable temp-dir tests. Artifact
    # integrity, serialization, real head execution and gradients stay real.
    monkeypatch.setattr(snapshot,'check_paths',lambda paths:[])
    torch.manual_seed(7)
    model=ContextDecisionModel(Backbone(),ContextDecisionHead(8,2,16),19).to(torch.bfloat16).eval()
    processor=tmp_path/'processor';processor.mkdir()
    (processor/'tokenizer.json').write_text(json.dumps(dict(added_tokens=[dict(id=19,content='<|jev_decision|>',special=True)])))
    (processor/'tokenizer_config.json').write_text('{}')
    contract=dict(base_id='a'*64,runtime_profile='fixture-bf16',request_version='fixture-v1',
        branch=dict(hidden_size=8,num_heads=2,ffn_size=16,residual_scale=.1,layers=1,
            position_encoding='inherited_backbone_states_no_extra_branch_encoding',dropout=0.,
            readout='decision_marker_itself',attention_scope='same_request_valid_causal_prefix_including_marker'),
        token=dict(id=19,text='<|jev_decision|>',row_trainable=False,registration_status='test_only'),
        output=dict(field='prob',activation='sigmoid',normalize_across_candidates=False),
        lora=snapshot.lora_contract(model.backbone))
    return model,processor,contract


def test_joint_roundtrip_probability_and_frozen_rows(tmp_path,monkeypatch):
    model,processor,contract=setup(tmp_path,monkeypatch)
    inputs=dict(input_ids=torch.tensor([[1,2,19,3,19]]),attention_mask=torch.ones(1,5,dtype=torch.long))
    positions=torch.tensor([[2,4]])
    with torch.no_grad():expected=model(inputs,positions).clone()
    original=model.backbone.embedding.weight.detach().clone()
    path=tmp_path/'policy';identity=snapshot.export_policy(path,model,processor,contract)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:p.add_(.25)
        model.backbone.embedding.weight[19].zero_()
    result=snapshot.load_policy(path,model,expected_policy_id=identity,
                               expected_contract=contract,expected_base_id='a'*64)
    with torch.no_grad():assert torch.equal(model(inputs,positions),expected)
    assert torch.equal(model.backbone.embedding.weight,original)
    assert result['optimizer_restored'] is False
    with pytest.raises(FileExistsError):snapshot.export_policy(path,model,processor,contract)


@pytest.mark.parametrize('fault',['weight','processor','extra','contract','base','policy'])
def test_mixed_or_corrupt_policy_rejected_without_mutation(tmp_path,monkeypatch,fault):
    model,processor,contract=setup(tmp_path,monkeypatch)
    path=tmp_path/'policy';identity=snapshot.export_policy(path,model,processor,contract)
    expected=copy.deepcopy(contract);base='a'*64
    if fault=='weight':
        with (path/'execution.pt').open('ab') as f:f.write(b'changed')
    elif fault=='processor':(path/'processor/tokenizer_config.json').write_text('{"changed":true}')
    elif fault=='extra':(path/'unexpected').write_text('extra')
    elif fault=='contract':expected['branch']['residual_scale']=.2
    elif fault=='base':base='b'*64
    elif fault=='policy':identity='c'*64
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    with pytest.raises(ValueError):
        snapshot.load_policy(path,model,expected_policy_id=identity,expected_contract=expected,expected_base_id=base)
    assert all(torch.equal(p,before[n]) for n,p in model.named_parameters())


def test_identity_includes_runtime_and_unexported_training_rejected(tmp_path,monkeypatch):
    model,processor,contract=setup(tmp_path,monkeypatch)
    a=snapshot.export_policy(tmp_path/'p1',model,processor,contract)
    modified=copy.deepcopy(contract);modified['runtime_profile']='another-runtime'
    b=snapshot.export_policy(tmp_path/'p2',model,processor,modified)
    assert a!=b
    model.backbone.embedding.requires_grad_(True)
    with pytest.raises(ValueError,match='frozen embedding'):
        snapshot.export_policy(tmp_path/'p3',model,processor,contract)


def test_nonfinite_marker_row_refused_before_publication(tmp_path,monkeypatch):
    model,processor,contract=setup(tmp_path,monkeypatch)
    with torch.no_grad():model.backbone.embedding.weight[19,0]=float('nan')
    with pytest.raises(ValueError,match='finite decision'):
        snapshot.export_policy(tmp_path/'bad',model,processor,contract)
    assert not (tmp_path/'bad').exists()


def test_same_weight_shapes_different_lora_scaling_rejected(tmp_path,monkeypatch):
    model,processor,contract=setup(tmp_path,monkeypatch)
    path=tmp_path/'policy';identity=snapshot.export_policy(path,model,processor,contract)
    model.backbone.q_proj.scaling['default']=2.
    with pytest.raises(ValueError,match='LoRA execution'):
        snapshot.load_policy(path,model,expected_policy_id=identity,expected_contract=contract,expected_base_id='a'*64)
