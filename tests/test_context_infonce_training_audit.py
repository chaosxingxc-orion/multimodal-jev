import copy
import json
from pathlib import Path

import pytest
import torch

from multimodal_jev.context_training_checkpoint import capture
from multimodal_jev.context_warmup import learning_rate
from tools.reports.audit_context_infonce_training import expected_contract,restore_shard,check_batch


def fixture():
    design=json.loads((Path(__file__).resolve().parents[1]/'configs/experiments/context-two-stage-v0.0.4.json').read_text())
    c=dict(design['stage1'],physical_devices=list(range(8)),microbatch=8,base_id='base',source_policy_id='initial',source_contract={},
           runtime_profile='profile',release_sha256='release',schedule_sha256='schedule',
           parent_schedule_sha256='parent',schedule_release_sha256='parent-release')
    p=torch.nn.Parameter(torch.tensor([.3,.4]));named=[('decision_head.projection.weight',p)]
    opt=torch.optim.AdamW([p],lr=learning_rate(c['lr_schedule'],0,160),betas=(.9,.999),
                          eps=1e-8,weight_decay=0.,foreach=False,fused=False)
    p.square().sum().backward();opt.step()
    state=capture(named,opt,rank=0,world_size=8,completed_updates=1,contract=expected_contract(c))
    return c,state


def test_accepts_actual_warmup_lr_and_restores_adam():
    c,state=fixture();result=restore_shard(state,c,rank=0,step=1)
    assert result['recovery']['exact_masters'] and result['recovery']['exact_adam']
    assert result['scope']['base_model_saved'] is False


@pytest.mark.parametrize('fault',['lr','objective','rank','cursor','base','moment'])
def test_rejects_inconsistent_saved_state(fault):
    c,state=fixture()
    if fault=='lr':state['optimizer']['param_groups'][0]['lr']=1e-4
    if fault=='objective':state['contract']['objective']['probability_coefficient']=.3
    if fault=='rank':state['rank']=1
    if fault=='cursor':state['completed_updates']=2
    if fault=='base':
        key=state['names'][0];state['names']=['backbone.embed_tokens.weight']
        state['parameters']={'backbone.embed_tokens.weight':state['parameters'][key]}
    if fault=='moment':state['optimizer']['state'][0]['exp_avg'][0]=float('nan')
    with pytest.raises(ValueError):restore_shard(state,c,rank=0,step=1)


def test_rejects_duplicate_rank_slots_before_loading_labels():
    batch=dict(policy_id='p',records=[{},{}],rank_indices=[[0],[0]])
    with pytest.raises(ValueError,match='permutation'):
        check_batch(batch,{}, {}, {},policy='p',width=2,world=2)


def test_replay_changes_paths_and_window_but_not_optimization_contract():
    from tools.reports.context_infonce_recovery import derive_config
    c,_=fixture()
    c.update(source_root='/old/source',schedule='/old/source/schedule',
             parent_schedule='/old/source/parent-schedule',pins={'/old/source/code.py':'code','/models/base':'base'},
             start_update=0,stop_update=2,resume_root=None)
    commit=dict(updates=1,rank_shards={f'rank-{i}/checkpoint-0001.pt':str(i) for i in range(8)})
    replay=derive_config(c,'/old','/new',commit)
    assert replay['start_update']==1 and replay['stop_update']==2 and replay['resume_root']=='/old/run'
    assert replay['pins']['/new/source/code.py']=='code' and replay['pins']['/models/base']=='base'
    assert expected_contract(c)==expected_contract(replay)
    assert c['start_update']==0 and c['source_root']=='/old/source'
    with pytest.raises(ValueError):derive_config(c,'/old','/new',dict(commit,updates=2))


def test_replay_comparison_checks_all_eight_ranks_and_optimizer(tmp_path):
    from tools.reports.context_infonce_recovery import compare
    for side in ('continuous','replay'):
        root=tmp_path/side;root.mkdir()
        (root/'batch-0001.json').write_text(json.dumps({'policy_id':'p','records':['same']}))
        (root/'checkpoint-0002-committed.json').write_text(json.dumps(dict(updates=2,contract={'loss':'infonce'})))
        torch.save({'head':torch.tensor([.2],dtype=torch.bfloat16)},root/'execution-0002.pt')
        (root/'policy-0002').mkdir()
        torch.save({'head':torch.tensor([.2],dtype=torch.bfloat16)},root/'policy-0002/execution.pt')
        for rank in range(8):
            d=root/f'rank-{rank}';d.mkdir()
            torch.save({'master':torch.tensor([.2]),'adam':torch.tensor([.01])},d/'checkpoint-0002.pt')
            (d/'restored.json').write_text(json.dumps(dict(exact_masters=True,exact_adam=True,completed_updates=1)))
    assert compare(tmp_path/'continuous',tmp_path/'replay')['rank_shards']==8
    p=tmp_path/'replay/rank-7/checkpoint-0002.pt'
    torch.save({'master':torch.tensor([.2]),'adam':torch.tensor([.02])},p)
    with pytest.raises(ValueError,match='rank-7/adam'):compare(tmp_path/'continuous',tmp_path/'replay')


def test_formal_uses_admitted_objective_full_budget_and_registered_curve():
    from tools.reports.context_infonce_formal import formal_config
    c,_=fixture();c.update(source_root='/old/source',schedule='/old/source/schedule',
                          parent_schedule='/old/source/parent-schedule',pins={'/old/source/code':'h'})
    design=json.loads((Path(__file__).resolve().parents[1]/'configs/experiments/context-two-stage-v0.0.4.json').read_text())
    full=formal_config(c,'/old','/new',design)
    assert full['stop_update']==160 and full['checkpoint_interval']==16 and full['reference_endpoint']==160
    assert full['evaluation_steps']==list(range(0,161,16))
    assert full['engineering_prefix'] is False and full['training_dispatch_allowed'] is True
    assert full['instruction_objective']['probability_coefficient']==0
    design['stage1']['instruction_objective']['temperature']=.5
    with pytest.raises(ValueError,match='admitted'):formal_config(c,'/old','/new',design)
