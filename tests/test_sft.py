import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from multimodal_jev.sft import select_training,initialization_identity,initialize_adapter
from multimodal_jev.task_schema import Candidate,DecisionSchema,JevExample


def test_training_selection_contains_complete_views_from_train(tmp_path):
    schema=DecisionSchema('s','choice','pick',(Candidate('a','A'),Candidate('b','B')))
    rows=[]
    for label in ['a','b']:
        for group in ['1','2']:
            for view in ['native','permutation','rebind','missing_candidate','noul.True','noul.False']:
                uid=label+group
                rows.append(JevExample(uid+view,'train','fixture.wav',schema,label,
                    {'source_group':uid,'variant':view}).to_record())
    path=tmp_path/'train.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    config={'train_manifest':str(path),'groups_per_class':1,'seed':17}
    selected=select_training(config)
    assert len(selected)==12 and len({x.metadata['source_group'] for x in selected})==2
    assert select_training(config)==selected


def continuation_fixture(tmp_path):
    from test_rl_preflight import fixture
    from multimodal_jev.baseline import file_hash
    cfg=fixture(tmp_path,adapted=True)
    config={'model_path':cfg['actor']['model_path'],'initial_adapter':cfg['initial_adapter'],
            'initial_checkpoint':cfg['initial_checkpoint'],'lora_rank':4,'lora_alpha':8}
    adapter=Path(config['initial_adapter'])/'adapter_config.json'
    adapter.write_text(json.dumps({'r':4,'lora_alpha':8}))
    receipt=Path(config['initial_checkpoint'])/'EXPORT.json'
    report=json.loads(receipt.read_text())
    report['adapter_files']['adapter_config.json']=file_hash(adapter)
    receipt.write_text(json.dumps(report))
    return config


def test_continuation_binds_initializer_and_original_base(tmp_path):
    config=continuation_fixture(tmp_path)
    identity=initialization_identity(config)
    assert identity['base_policy_id']!=identity['policy_id']
    assert identity['adapter_files']['adapter_model.safetensors']
    assert identity['optimizer_start'].startswith('fresh')
    assert initialization_identity({}) is None
    with pytest.raises(ValueError,match='together'):
        initialization_identity({**config,'initial_checkpoint':None})
    with pytest.raises(ValueError,match='rank/alpha'):
        initialization_identity({**config,'lora_rank':8})
    (Path(config['initial_adapter'])/'adapter_model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError,match='adapter/base'):
        initialization_identity(config)


def test_continuation_loads_trainable_saved_adapter_without_reinitializing(monkeypatch):
    calls=[];base=object();adapted=object()
    def load(model,path,**kwargs):
        calls.append((model,path,kwargs));return adapted
    def forbidden(*args,**kwargs):raise AssertionError('new LoRA initialization is forbidden')
    monkeypatch.setitem(sys.modules,'peft',SimpleNamespace(
        PeftModel=SimpleNamespace(from_pretrained=load),LoraConfig=forbidden,get_peft_model=forbidden))
    assert initialize_adapter(base,{'initial_adapter':'shared-adapter'}) is adapted
    assert calls==[(base,'shared-adapter',{'is_trainable':True})]
