from types import SimpleNamespace

import pytest
import torch

from multimodal_jev.context_reference_builder import build
from multimodal_jev.context_reference_dataset import ReferenceDataset,digest
from multimodal_jev.context_logits_wire import WIRE
from multimodal_jev.encoded_audio_cache import canonical
from multimodal_jev.span_request_identity import tensor_identity


def setup(monkeypatch):
    import multimodal_jev.context_reference_dataset as dataset
    import multimodal_jev.local_checkpoint as checkpoint
    monkeypatch.setattr(dataset,'check_paths',lambda _:None)
    monkeypatch.setattr(checkpoint,'check_paths',lambda _:None)
    identity=dict(schema='jev-reference-dataset-v1',policy_id='a'*64,base_id='b'*64,contract={'base_id':'b'*64},
        runtime_identity=dict(image_sha256='1'*64,source_manifest_sha256='2'*64,processor_sha256='3'*64,
                              profile='test',output_protocol=WIRE),release_sha256='c'*64,candidate_ids=['a','b'])
    candidates=[dict(id='a',definition='first'),dict(id='b',definition='second')]
    assets={f's{i}':dict(audio=f'/audio/{i%3}.wav',audio_sha256=str(i%3)*64) for i in range(4)}
    class Engine:
        bundle={'policy_id':identity['policy_id']}
        runtime_identity=identity['runtime_identity']
        calls=[]
        fault=None
        def score_many(self,rows,*,audio_token_id):
            self.calls.append(len(rows));result=[]
            for row in rows:
                binding=dict(policy_id=self.bundle['policy_id'],positions=row['positions'][0].tolist(),
                    inputs={k:tensor_identity(v) for k,v in sorted(row['inputs'].items())})
                z=torch.tensor([.5,-.5]);p=z.sigmoid()
                result.append(dict(request_id=digest(binding),request_identity=binding,
                    policy_id=self.bundle['policy_id'],prob=p.tolist(),logits=z.tolist(),output_protocol=WIRE))
            if self.fault=='order':result.reverse()
            if self.fault=='policy':self.bundle={'policy_id':'f'*64}
            if self.fault=='missing':result.pop()
            return result
    engine=Engine()
    def preparer(job):
        request,h=job;i=int(h[0])
        x=dict(input_ids=torch.tensor([[i,99,99]]),attention_mask=torch.ones(1,3,dtype=torch.long),
               input_features=torch.ones(1,2,3)*i,input_features_mask=torch.ones(1,3,dtype=torch.long))
        return SimpleNamespace(inputs=x,positions=torch.tensor([[1,2]]),cache_hit=True,
                               request=canonical(request.model_input()))
    return dict(assets=assets,candidates=candidates,requirements=[],preparer=preparer,engine=engine,
                identity=identity,macro_queries=2,audio_token_id=7,producer={'scope':'test'})


def test_full_sample_mapping_duplicate_inputs_and_complete_macros(tmp_path,monkeypatch):
    kw=setup(monkeypatch);result=build(tmp_path/'reference',**kw)
    assert result['recordings']==4 and result['distinct_inputs']==3
    assert kw['engine'].calls==[2,1]
    assert result['population']['s0']==result['population']['s3']
    reader=ReferenceDataset(tmp_path/'reference',manifest_sha256=result['manifest_sha256'],
        expected_identity=result['identity'],expected_population=result['population'])
    assert len(reader.population)==4


@pytest.mark.parametrize('fault',['order','policy','missing','labels','candidate'])
def test_bad_producer_or_population_never_publishes(tmp_path,monkeypatch,fault):
    kw=setup(monkeypatch)
    if fault=='labels':kw['assets']['s0']['intent']='a'
    elif fault=='candidate':kw['candidates'].reverse()
    else:kw['engine'].fault=fault
    with pytest.raises(ValueError):build(tmp_path/'reference',**kw)
    assert not (tmp_path/'reference').exists()


@pytest.mark.parametrize('fault',[None,'prefix','release','base','auxiliary','stage','world','contract'])
def test_reference_requires_actual_full_stage1_endpoint(fault):
    import json
    from pathlib import Path
    from tools.reports.build_context_reference_dataset import validate_endpoint
    s=json.loads((Path(__file__).resolve().parents[1]/'configs/experiments/context-two-stage-v0.0.11.json').read_text())['stage1']
    c=dict(stage1_endpoint=318,release_sha256='release',base_id='base',expected_population=40513)
    bundle=dict(policy_id='stage1',contract={'base_id':'base'})
    checkpoint=dict(updates=318,joint_policy={'policy_id':'stage1'},contract=dict(batch_config=s['batch_config'],
        objective=s['instruction_objective'],release_sha256='release',base_id='base',
        stage='instruction',world_size=4,source_contract={'base_id':'base'}))
    if fault=='prefix':checkpoint['updates']=c['stage1_endpoint']=2
    if fault in ('release','base'):checkpoint['contract'][fault+'_id' if fault=='base' else 'release_sha256']='other'
    if fault=='auxiliary':checkpoint['contract']['objective']['probability_coefficient']=1.
    if fault=='stage':checkpoint['contract']['stage']='task'
    if fault=='world':checkpoint['contract']['world_size']=8
    if fault=='contract':checkpoint['contract']['source_contract']={'changed':True}
    if fault is None:validate_endpoint(checkpoint,c,bundle)
    else:
        with pytest.raises(ValueError,match='full InfoNCE'):validate_endpoint(checkpoint,c,bundle)
