"""The ordinary endpoint executes the configured audio provider, also used for training."""
import json
from pathlib import Path

import pytest
import torch

from multimodal_jev.baseline import file_hash
from multimodal_jev.model_provider import ABI, cache_namespace, require_same_provider, resolve_provider
from tools.reports import evaluate_tpo_endpoints as endpoint
from tools.reports.prepare_tpo_evaluation import evaluation_cache


@pytest.fixture
def external(tmp_path, monkeypatch):
    module = tmp_path/'fixture_speech_encoder.py'
    module.write_text("""
import torch
ABI = 'jev-audio-model-v1'
class Speech:
    runtime_profile = 'provider:fixture-v1'
    def __init__(self, options):
        self.scale = options['scale']
    def forward(self, model, items, **kw):
        return torch.stack([r['inputs']['input_features'].sum()*model for r in items])[:,None]
    def load_reference(self, config, bundle, output):
        return torch.tensor(self.scale), 'cpu', None, None
    def load_actor(self, *args):
        return torch.nn.Parameter(torch.tensor(self.scale))
    def processor(self, path): return None
    def collation(self,*args): return {}
    def unavailable(self,*args): raise AssertionError('unexpected role')
    choice_compiler = encode = initialize_policy = initialize_actor_runtime = adapter_transition = unavailable
create = Speech
""")
    monkeypatch.syspath_prepend(str(tmp_path))
    return dict(abi=ABI,module=module.stem,sha256=file_hash(module),options={'scale':1.25})


def test_train_and_formal_endpoint_use_bound_audio_provider(tmp_path, monkeypatch, external):
    config = dict(model_provider=external,base_id='base',model='weights',
                  source_contract={'runtime_profile':'provider:fixture-v1'},source_policy='checkpoint',
                  source_policy_id='initial')
    parent = tmp_path/'parent.json'
    parent.write_text(json.dumps(config))
    audio = [{'id':'a','inputs':{'input_features':torch.tensor([1.,2.])}},
             {'id':'b','inputs':{'input_features':torch.tensor([-.4])}}]
    # Supply only IO fixtures; real provider resolution, actor gradient, batched
    # and singleton evaluation forwards and prediction publication all execute.
    provider = resolve_provider(config)
    param = provider.load_actor()
    logits = provider.forward(param,audio)
    logits.sum().backward()
    torch.testing.assert_close(param.grad, torch.tensor(2.6))
    monkeypatch.setattr(endpoint,'checkpoint_identity',lambda _:('base',{}))
    monkeypatch.setattr(endpoint,'read_execution_bundle',lambda *a,**k:{'processor':'processor'})
    monkeypatch.setattr(endpoint,'processor_cache_identity',lambda _:'sha')
    monkeypatch.setattr(endpoint,'load_policy',lambda *a,**k:None)
    monkeypatch.setattr(endpoint,'load',lambda c:(audio,['x'],lambda row:(row,None)))
    # pytest tmp_path may be tmpfs; the endpoint's hot-storage policy is tested separately.
    monkeypatch.setattr('multimodal_jev.local_checkpoint.check_paths',lambda _:None)
    output = tmp_path/'evaluation'
    endpoint.run(dict(output=str(output),parent=str(parent),parent_sha=file_hash(parent),
                      endpoint={'cache':{'model_provider':external,'dataset':'demo','processor':{}}},
                      devices=[0], batch=2, shard=0, shards=1, arms=['stage1']))
    predictions=json.loads((output/'stage1.json').read_text())
    assert predictions['policy_id']=='initial'
    assert [r['id'] for r in predictions['records']]==['a','b']
    torch.testing.assert_close(torch.tensor([r['prob'] for r in predictions['records']]),logits.detach().sigmoid())


def test_incompatible_binding_fails_closed(external):
    c=dict(model_provider=external)
    assert cache_namespace(c)!=cache_namespace(dict(model_provider=dict(external,options={'scale':2})))
    assert cache_namespace({}) is None
    with pytest.raises(ValueError,match='differs between'):
        require_same_provider(c,{})
    with pytest.raises(ValueError,match='source differs'):
        resolve_provider(dict(model_provider=dict(external,sha256='0'*64)))
    with pytest.raises(ValueError,match='incompatible'):
        resolve_provider(dict(c,contract={'runtime_profile':'other'}))
    with pytest.raises(ValueError,match='binding'):
        resolve_provider({'model_provider':None})


def test_formal_eval_cache_can_be_published_without_a_historical_package(tmp_path, external):
    root = tmp_path/'cache'
    root.mkdir()
    parent = dict(model_provider=external)
    encoding = dict(decision_token_id=7, audio_token_id=3, max_tokens=128)
    cache = dict(parent, dataset='source_TEST', output=str(root), encoding=encoding,
                 processor={'sha256':'processor'}, seal={'sha256':'seal'})
    cfg = tmp_path/'cache.json'
    cfg.write_text(json.dumps(cache))
    manifest = root/'MANIFEST.json'
    manifest.write_text(json.dumps(dict(schema='jev-native-seal-encoded-cache-v1', encoding=encoding,
                                       seal_sha256='seal', model_provider_sha256=cache_namespace(parent),
                                       processor_sha256='processor', recordings=2)))
    verified = root/'VERIFIED.json'
    verified.write_text(json.dumps(dict(status='full_stable_prefix_cache_verified',
                                        manifest_sha256=file_hash(manifest), recordings=2)))
    source = {'cache_config':{'path':str(cfg), 'sha256':file_hash(cfg)}}
    assert evaluation_cache(source, 'source_TEST', parent, None) == cache
    with pytest.raises(ValueError, match='differs between'):
        evaluation_cache(source, 'source_TEST', {}, None)
    manifest.write_text(manifest.read_text()+'\n')
    with pytest.raises(ValueError, match='independent receipt'):
        evaluation_cache(source, 'source_TEST', parent, None)
