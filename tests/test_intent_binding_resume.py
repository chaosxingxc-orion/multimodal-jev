import copy
from pathlib import Path
import pytest
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.intent_binding_resume import validate_config,validate_lineage,authenticate,SCHEMA
from multimodal_jev.span_candidate_intervention import VARIANTS


def configs(tmp_path):
    previous=dict(schema=SCHEMA,binding_release='/binding',binding_manifest_sha256='manifest',model_path='/base',
        initial_adapter='/initial',acceptance_root='/initial-gate',image='fixed-image',arm='P',source_batch=8,
        views_per_source=4,global_requests=32,world_size=4,variants=list(VARIANTS),seed=317,step=0,
        lora_rank=4,lora_alpha=8,lora_execution_dtype='bfloat16',
        runtime_profile='qwen3-asr-bf16-hf-order-short-audio-v2',lr=1e-4,pins={})
    root=tmp_path/'training';suite=tmp_path/'suite'
    current=dict(previous,step=1,previous_root=str(root),previous_suite=str(suite),
        initial_adapter=str(root/'run/adapter'),acceptance_root=str(suite),pins={})
    return previous,current


@pytest.mark.parametrize('key,value',[('source_batch',32),('views_per_source',1),('global_requests',8),
    ('world_size',1),('world_size',4.),('step',True),('step',-1),('seed',True),('variants',['native']),
    ('lora_execution_dtype','float32'),('runtime_profile','old'),('schema','native'),('arm','SFT'),
    ('lora_rank',True),('lr',float('nan')),('initial_optimizer_checkpoint','foreign')])
def test_binding_configuration_rejects_budget_or_precision_drift(tmp_path,key,value):
    previous,current=configs(tmp_path);validate_config(previous);validate_config(current)
    previous[key]=value
    with pytest.raises(ValueError):validate_config(previous)


@pytest.mark.parametrize('key,value',[('binding_manifest_sha256','other'),('binding_release','/other'),
    ('arm','D'),('seed',318),('step',2),('image','other'),('lora_alpha',16),('lr',.0002),
    ('adam_betas',[0.,.999]),('initial_adapter','/other'),('acceptance_root','/other')])
def test_resume_does_not_change_objective_or_optimizer(tmp_path,key,value):
    previous,current=configs(tmp_path);validate_lineage(previous,current);current[key]=value
    with pytest.raises(ValueError):validate_lineage(previous,current)


def fixture(tmp_path,monkeypatch):
    import multimodal_jev.intent_binding_resume as module
    import multimodal_jev.intent_binding_gate as gate_module
    previous,current=configs(tmp_path);root=Path(current['previous_root']);suite=Path(current['previous_suite'])
    root.mkdir();suite.mkdir();files={}
    selection=dict(source_ids=['s'+str(i) for i in range(8)],view_ids=['v'+str(i) for i in range(32)],
        gate_ids=['g'+str(i) for i in range(32)],source_uses=8,view_uses=32,request_weight_sum=8.,
        step=0,seed=317,arm='P',binding_manifest_sha256='manifest')
    for rank in range(4):
        for name in ('optimizer-local.pt','result.json','terminal.json','selection.json','before.json','after.json'):
            p=root/'run'/f'rank-{rank}'/name;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text('synthetic fixture '+str(rank)+' '+name);files[str(p)]=file_hash(p)
    atomic_json(root/'run/selection.json',selection);files[str(root/'run/selection.json')]=file_hash(root/'run/selection.json')
    atomic_json(root/'config.json',previous)
    atomic_json(suite/'config.json',dict(training_root=str(root),adapter=current['initial_adapter'],model='/base',
        runtime_profile=current['runtime_profile'],base_id='base',adapter_id='adapter',population=dict(recordings=89,requests=712)))
    atomic_json(suite/'verified.json',dict(schema='intent-binding-update-verified-v1',passed=True,root=str(root),suite=str(suite),
        updates=1,arm='P',binding_manifest_sha256='manifest',full_gate_requests=712,source_uses=8,view_uses=32,
        exact_export_reload=True,adapter_id='adapter',files=files))
    atomic_json(suite/'ACCEPTED.json',dict(passed=True,adapter_id='adapter',verified_sha256=file_hash(suite/'verified.json')))
    atomic_json(suite/'controller-terminal.json',dict(exit_code=0))
    current['pins']={**files,**{str(p):file_hash(p) for p in [root/'config.json',*suite.glob('*.json')]}}
    monkeypatch.setattr(module,'expected_selection',lambda cfg:([],copy.deepcopy(selection)))
    monkeypatch.setattr(gate_module,'authenticate_artifacts',lambda *args:(None,None,dict(passed=True,requests=712,actual_exit_observed=False)))
    return current


def test_complete_binding_artifacts_allow_lineage_validation_only(tmp_path,monkeypatch):
    result=authenticate(fixture(tmp_path,monkeypatch))
    assert result['previous_updates']==1 and result['source_uses']==8 and result['view_uses']==32
    assert not result['gate']['actual_exit_observed']


@pytest.mark.parametrize('defect',['native','budget','missing_rank','shard_changed','unpinned','schedule','phase','full_gate'])
def test_binding_resume_rejects_false_acceptance(tmp_path,monkeypatch,defect):
    import json
    cfg=fixture(tmp_path,monkeypatch);root=Path(cfg['previous_root']);suite=Path(cfg['previous_suite'])
    def rewrite(path,data):atomic_json(path,data);cfg['pins'][str(path)]=file_hash(path)
    if defect in ('native','budget','missing_rank'):
        path=suite/'verified.json';v=json.loads(path.read_text())
        if defect=='native':v['full_gate_requests']=178
        elif defect=='budget':v['source_uses']=32
        else:v['files'].pop(str(root/'run/rank-3/optimizer-local.pt'))
        rewrite(path,v);a=json.loads((suite/'ACCEPTED.json').read_text());a['verified_sha256']=file_hash(path);rewrite(suite/'ACCEPTED.json',a)
    elif defect=='shard_changed':(root/'run/rank-3/optimizer-local.pt').write_text('changed')
    elif defect=='unpinned':cfg['pins'].pop(str(root/'run/rank-3/optimizer-local.pt'))
    elif defect=='schedule':
        p=root/'run/selection.json';s=json.loads(p.read_text());s['source_ids'].reverse();rewrite(p,s)
    elif defect=='phase':rewrite(suite/'controller-terminal.json',dict(exit_code=1))
    else:
        import multimodal_jev.intent_binding_gate as gate
        def reject(*args):raise ValueError('full gate rejected')
        monkeypatch.setattr(gate,'authenticate_artifacts',reject)
    with pytest.raises(ValueError):authenticate(cfg)
