"""CPU synthetic artifact integration of the new verifier; no NPU claim."""
import copy
import json
import math
from pathlib import Path
import sys
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
import verify_intent_binding_npu as module
from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.intent_optimizer import optimizer_kwargs
from test_intent_binding_resume import configs
from test_intent_binding_inputs import rows


def fixture(tmp_path,monkeypatch,step=0):
    root=tmp_path/'train';suite=tmp_path/'suite';root.mkdir();suite.mkdir()
    cfg,_=configs(tmp_path);cfg.update(arm='P',step=step)
    if step:cfg.update(previous_root=str(tmp_path/'prior'),previous_suite=str(tmp_path/'prior-suite'))
    selected=rows()[:32]
    selection=dict(source_ids=[str(i) for i in range(8)],view_ids=[r['id'] for r in selected],
        gate_ids=[r['gate_id'] for r in selected],source_uses=8,view_uses=32,request_weight_sum=16.,
        step=step,seed=317,arm='P',binding_manifest_sha256='manifest')
    gate_audit=dict(passed=True,requests=712,actual_exit_observed=False)
    selection['gate']=gate_audit
    behavior={};refs={}
    for item in selected:
        probs=[[-math.log(len(q.values))]*len(q.values) for q in item['request'].questions]
        behavior[item['gate_id']]=dict(request_id='old/'+item['id'],candidate_logprobs=probs)
        refs[item['gate_id']]=dict(candidate_logprobs=probs,
            request_identity=dict(request=json.loads(json.dumps(item['request'].model_input()))))
    atomic_json(root/'config.json',cfg);atomic_json(root/'source-manifest.json',{})
    atomic_json(root/'launch.json',dict(image=cfg['image'],physical_devices=[4,5,6,7],
        config_sha256=file_hash(root/'config.json'),source_manifest_sha256=file_hash(root/'source-manifest.json')))
    (root/'run').mkdir();atomic_json(root/'run/selection.json',selection)
    supervision=[dict(request_id=behavior[r['gate_id']]['request_id'],targets=[r['target']['targets']],
        supervised=[r['target']['supervised']],span_weights=torch.tensor([r['target']['span_weights']]).tolist(),
        request_weight=r['target']['request_weight']) for r in selected]
    atomic_json(root/'run/supervision.json',supervision)
    artifact=tmp_path/'inputs.json';atomic_json(artifact,{})
    scfg=dict(adapter=str(root/'run/adapter'),model=cfg['model_path'],runtime_profile=cfg['runtime_profile'],
        inputs=str(artifact),pins={str(artifact):file_hash(artifact)},training_root=str(root),
        population=dict(recordings=89,requests=712),base_id='base',adapter_id='new')
    atomic_json(suite/'config.json',scfg);atomic_json(suite/'source-manifest.json',{})
    atomic_json(suite/'launch.json',dict(image=cfg['image'],physical_devices=[4],
        config_sha256=file_hash(suite/'config.json'),source_manifest_sha256=file_hash(suite/'source-manifest.json')))
    for name in ('hf','vllm'):
        (suite/'run'/name).mkdir(parents=True)
        atomic_json(suite/'run'/name/'result.json',dict(passed=True,base_id='base',adapter_id='new'))
        atomic_json(suite/'run'/name/'terminal.json',dict(exit_code=0))
    atomic_json(suite/'run/terminal.json',dict(exit_code=0))
    atomic_json(suite/'run/READY.json',dict(passed=True,requests=712,base_id='base',adapter_id='new',result_sha256=file_hash(suite/'run/vllm/result.json')))
    admission=dict(passed=True,world_size=4,global_requests=32,global_effective_weight=16.,rank_effective_weights=[4.]*4)
    for rank in range(4):
        out=root/'run'/f'rank-{rank}';out.mkdir();local=selected[rank*8:(rank+1)*8];ids=[r['id'] for r in local]
        before=[[p for span in behavior[r['gate_id']]['candidate_logprobs'] for p in span] for r in local]
        old=torch.tensor(before,dtype=torch.float32).tolist()
        atomic_json(out/'before.json',dict(old=old,actor=old,admission=admission))
        atomic_json(out/'after.json',[dict(id=r['id'],previous_request_id=behavior[r['gate_id']]['request_id'],
            candidate_logprobs=refs[r['gate_id']]['candidate_logprobs']) for r in local])
        atomic_json(out/'selection.json',dict(selection,rank=rank,local_view_ids=ids))
        restore=dict(previous_updates=step,exact_masters=True,exact_adam=True) if step else None
        atomic_json(out/'result.json',dict(passed=True,rank=rank,world_size=4,updates=step+1,arm='P',
            actor_sha256=module.ACTOR_SHA256,base_id='base',initial_policy_id='base:old',adapter_id='new',
            view_ids=ids,source_ids=selection['source_ids'][rank*2:(rank+1)*2],gate_ids=[r['gate_id'] for r in local],
            frozen_base_unchanged=True,cross_rank_export_exact=True,admission=admission,
            execution_dtypes={str(i):['torch.bfloat16','torch.bfloat16'] for i in range(112)},restored=restore))
        atomic_json(out/'terminal.json',dict(exit_code=0))
        names=[str(i) for i in range(112)]
        state=dict(rank=rank,world_size=4,updates=step+1,names=names,parameters={n:torch.ones(2) for n in names},
            optimizer=dict(param_groups=[dict(optimizer_kwargs(cfg),params=list(range(112)))],
                state={i:dict(step=torch.tensor(float(step+1)),exp_avg=torch.ones(2),exp_avg_sq=torch.ones(2)) for i in range(112)}))
        torch.save(state,out/'optimizer-local.pt')
        if step:
            prior=copy.deepcopy(state);prior['updates']=step
            for s in prior['optimizer']['state'].values():s['step']=torch.tensor(float(step))
            dest=Path(cfg['previous_root'])/'run'/f'rank-{rank}';dest.mkdir(parents=True)
            torch.save(prior,dest/'optimizer-local.pt')
            torch.save({k:prior[k] for k in ('names','parameters','optimizer')},out/'restored-local.pt')
            atomic_json(out/'optimizer-resume.json',restore)
    lineage=dict(previous_updates=step)
    if step:atomic_json(root/'run/resume-lineage.json',lineage)
    monkeypatch.setattr(module,'checkpoint_identity',lambda p:('new' if Path(p)==root/'run/adapter' else 'base' if str(p)==cfg['model_path'] else 'old',{}))
    monkeypatch.setattr(module,'expected_selection',lambda c:(selected,{k:v for k,v in selection.items() if k!='gate'}))
    monkeypatch.setattr(module,'load',lambda *args:([],{}))
    monkeypatch.setattr(module,'verify_records',lambda *args:([refs,refs],dict(maximum_logprob_error=0.)))
    monkeypatch.setattr(module,'authenticate_artifacts',lambda *args:(None,[behavior,behavior],gate_audit))
    import multimodal_jev.intent_binding_resume as resume
    monkeypatch.setattr(resume,'authenticate',lambda c:lineage)
    return root,suite


@pytest.mark.parametrize('step',[0,1])
def test_complete_first_and_resumed_update_artifacts(tmp_path,monkeypatch,step):
    root,suite=fixture(tmp_path,monkeypatch,step)
    result=module.verify(root,suite)
    assert result['schema']=='intent-binding-update-verified-v1'
    assert result['source_uses']==8 and result['view_uses']==32 and result['full_gate_requests']==712
    assert result['updates']==step+1 and result['exact_export_reload']


@pytest.mark.parametrize('defect',['target','reward','old','actor','after','view','source','gate_id','rank_selection',
    'dtype','adam','restore','phase','weight'])
def test_false_update_is_rejected(tmp_path,monkeypatch,defect):
    root,suite=fixture(tmp_path,monkeypatch,1);out=root/'run/rank-3'
    if defect in ('adam','restore'):
        path=out/('optimizer-local.pt' if defect=='adam' else 'restored-local.pt');state=torch.load(path,weights_only=True)
        if defect=='adam':state['optimizer']['state'][0]['exp_avg_sq'][0]=-1
        else:state['parameters']['0'][0]+=.25
        torch.save(state,path)
    else:
        name={'target':'supervision','reward':'supervision','old':'before','actor':'before','after':'after',
            'rank_selection':'selection','phase':'terminal'}.get(defect,'result')
        path=(root/'run' if name=='supervision' else out)/(name+'.json');data=json.loads(path.read_text())
        if defect=='target':data[-1]['targets'][0][-1]=(data[-1]['targets'][0][-1]+1)%60
        elif defect=='reward':data[-1]['span_weights'][0][0]=1.
        elif defect=='old':data['old'][-1][0]+=.1
        elif defect=='actor':data['actor'][-1][0]+=.1
        elif defect=='after':data[-1]['candidate_logprobs'][-1][0]+=.1
        elif defect=='view':data['view_ids'].reverse()
        elif defect=='source':data['source_ids'].reverse()
        elif defect=='gate_id':data['gate_ids'].reverse()
        elif defect=='rank_selection':data['local_view_ids'].reverse()
        elif defect=='dtype':data['execution_dtypes']['0']=['torch.float32','torch.float32']
        elif defect=='phase':data['exit_code']=1
        elif defect=='weight':data['admission']['global_effective_weight']=32.
        atomic_json(path,data)
    with pytest.raises(ValueError):module.verify(root,suite)
