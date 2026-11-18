import copy
import json

import pytest
import torch

from multimodal_jev.objective_plugin import ObjectivePlugin
from multimodal_jev.objective_trace import pack,unpack,digest,replay,audit_update,FiniteTraceWriter
from tests.test_objective_plugin import ROOT,binding,batch,surrogate


def captured(plugin=None):
    plugin=plugin or ObjectivePlugin(binding(ROOT/'extensions/stage2_objective.py'))
    args=batch();rows=[]
    loss,metrics=plugin.hybrid_loss(*args,global_real_requests=2,trace_observer=rows.append)
    return plugin,args,rows[0],loss


def test_trace_roundtrip_reward_loss_and_gradient(surrogate):
    plugin,args,row,loss=captured()
    restored=json.loads(json.dumps(row))
    assert replay(restored,plugin)['status']=='objective_formula_replayed'
    baseline,_=plugin.hybrid_loss(*args,global_real_requests=2)
    g0,=torch.autograd.grad(loss,args[0],retain_graph=True)
    g1,=torch.autograd.grad(baseline,args[0])
    assert torch.equal(g0,g1)
    assert pack(unpack(row['inputs']))==row['inputs']


@pytest.mark.parametrize('fault',['digest','reward','loss','metric','identity'])
def test_replay_rejects_changed_evidence_even_with_recomputed_digest(surrogate,fault):
    plugin,args,row,loss=captured()
    if fault=='reward':
        value=unpack(row['rewards']);value[0]['existence'][0,0]+=0.5;row['rewards']=pack(value)
    if fault=='loss' or fault=='digest':row['loss']+=1
    if fault=='metric':
        value=unpack(row['metrics']);value['rl']+=1;row['metrics']=pack(value)
    if fault=='identity':row['objective']=pack(dict(plugin.identity,options={'different':1}))
    if fault!='digest':row['sha256']=digest({k:v for k,v in row.items() if k!='sha256'})
    with pytest.raises(ValueError):replay(row,plugin)


def test_actual_custom_reward_is_recorded(surrogate,tmp_path):
    path=tmp_path/'objective.py'
    path.write_text((ROOT/'extensions/stage2_objective.py').read_text().replace(
        "branch['rewards'].clone()","0.2 + 0.7 * branch['rewards'].clone()"))
    plugin,args,row,loss=captured(ObjectivePlugin(binding(path)))
    expected=0.2+0.7*args[1][0]['branches']['existence']['rewards']
    assert torch.equal(unpack(row['rewards'])[0]['existence'],expected)
    replay(row,plugin)


def test_writer_commits_once_and_rejects_wrong_update(surrogate,tmp_path,monkeypatch):
    import multimodal_jev.local_runtime_storage as storage
    import multimodal_jev.local_checkpoint as checkpoint
    monkeypatch.setattr(storage,'check_paths',lambda _:None)
    monkeypatch.setattr(checkpoint,'check_paths',lambda _:None)
    _,_,row,_=captured();writer=FiniteTraceWriter(tmp_path/'traces')
    with pytest.raises(ValueError,match='outside'):writer(row)
    writer.begin(1)
    with pytest.raises(ValueError,match='another update'):writer(row)
    with pytest.raises(ValueError,match='missing'):writer.finish()
    writer=FiniteTraceWriter(tmp_path/'traces');writer.begin(0);writer(row)
    path=writer.finish();assert json.loads(open(path).read())['rows']==[row]
    writer.begin(0);writer(row)
    with pytest.raises(FileExistsError):writer.finish()


def update_fixture():
    plugin,args,_,_=captured();z,records,identities,ref,real=args
    ranks=[];docs=[]
    for i in range(3):
        g=dict(dual_identity=identities[i],dual_credit=records[i],reference_probability=ref[i],
               real_request=bool(real[i]),dual_options=dict(arm='joint'))
        ranks.append([g]);rows=[]
        plugin.hybrid_loss(z[i:i+1],records[i:i+1],identities[i:i+1],ref[i:i+1],real[i:i+1],
                           global_real_requests=2,world_size=3,trace_observer=rows.append)
        docs.append(dict(schema='jev-objective-update-traces-v1',update=0,rows=rows))
    return plugin,ranks,docs


def test_all_rank_population_and_padding_bound_to_separate_groups(surrogate):
    plugin,ranks,docs=update_fixture()
    result=audit_update(docs,ranks,plugin,update=0,microbatch=1)
    assert result['queries']==3 and result['real_queries']==2


@pytest.mark.parametrize('fault',['missing','duplicate','reference','padding','denominator','rank','order'])
def test_complete_audit_rejects_population_substitution(surrogate,fault):
    plugin,ranks,docs=update_fixture()
    if fault=='missing':docs[0]['rows']=[]
    elif fault=='duplicate':docs[1]=copy.deepcopy(docs[0])
    elif fault=='order':docs.reverse()
    else:
        row=docs[0]['rows'][0];args=unpack(row['inputs'])
        if fault=='reference':args['reference_probability']*=.9
        if fault=='padding':args['real_requests'][0]=False
        if fault=='denominator':args['global_real_requests']=3
        if fault=='rank':args['world_size']=1
        row['inputs']=pack(args);row['sha256']=digest({k:v for k,v in row.items() if k!='sha256'})
    with pytest.raises(ValueError):audit_update(docs,ranks,plugin,update=0,microbatch=1)


def test_configured_auditor_replays_bound_files_and_rejects_substitution(tmp_path,surrogate):
    import hashlib
    from tools.reports.audit_objective_traces import audit
    plugin,ranks,docs=update_fixture()
    groups=tmp_path/'groups.pt';torch.save([g for rank in ranks for g in rank],groups)
    h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    traces=[]
    for rank,doc in enumerate(docs):
        path=tmp_path/f'rank-{rank}.json';path.write_text(json.dumps(doc))
        traces.append(dict(rank=rank,path=str(path),sha256=h(path)))
    c=dict(objective_plugin=binding(ROOT/'extensions/stage2_objective.py'),world_size=3,
           local_batch=1,microbatch=1,loss_atol=0.,output=str(tmp_path/'audit.json'),
           updates=[dict(update=0,groups=dict(path=str(groups),sha256=h(groups)),traces=traces)])
    assert audit(c,{})['updates'][0]['queries']==3
    c['updates'][0]['traces'][0]['sha256']='0'*64
    with pytest.raises(ValueError,match='evidence changed'):audit(c,{})


def test_actor_records_actual_registered_loss_and_preserves_gradient(tmp_path,surrogate,monkeypatch):
    from types import SimpleNamespace
    from tools.reports import context_dual_actor_candidate as original
    from tools.reports.context_plugin_trainer import load_actor
    import multimodal_jev.local_runtime_storage as storage
    import multimodal_jev.local_checkpoint as checkpoint
    monkeypatch.setattr(storage,'check_paths',lambda _:None)
    monkeypatch.setattr(checkpoint,'check_paths',lambda _:None)
    monkeypatch.setattr(original.dist,'get_world_size',lambda:1)
    config=binding(ROOT/'extensions/stage2_objective.py');actor=load_actor(ROOT,config)
    args=batch();z,records,identities,refs,real=args
    payload=[dict(dual_credit=r,dual_identity=i,reference_probability=ref,real_request=bool(mask),
                  global_real_requests=2,dual_options=dict(arm='joint'))
             for r,i,ref,mask in zip(records,identities,refs,real,strict=True)]
    cfg=SimpleNamespace(policy_loss=SimpleNamespace(loss_mode=original.MODE),clip_ratio=.2,
                        ppo_mini_batch_size=3,ppo_micro_batch_size_per_gpu=3)
    logp=torch.zeros(3,6);token=actor._active.set(dict(consumed=False,log_probs=logp,logits=z,payload=payload))
    try:
        actor.begin_objective_trace(tmp_path/'trace',0)
        loss,_=actor.registered_loss(None,logp,None,None,config=cfg)
        loss.backward();path=actor.finish_objective_trace()
    finally:actor._active.reset(token)
    doc=json.loads(open(path).read())
    result=audit_update([doc],[payload],ObjectivePlugin(config),update=0,microbatch=3)
    assert result['microbatches']==1 and not z.grad[2].any()


@pytest.mark.parametrize('target',['rewards','reference'])
def test_trace_rejects_in_place_mutation_by_plugin(tmp_path,surrogate,target):
    path=tmp_path/'mutating.py'
    mutation=("    rewards[0]['existence'][0,0] += .25\n" if target=='rewards'
              else "    ctx.reference_probability[0,0] = .4\n")
    path.write_text((ROOT/'extensions/stage2_objective.py').read_text().replace(
        '    return ctx.baseline_loss(rewards)',mutation+'    return ctx.baseline_loss(rewards)'))
    with pytest.raises(ValueError,match='mutated'):
        captured(ObjectivePlugin(binding(path)))
