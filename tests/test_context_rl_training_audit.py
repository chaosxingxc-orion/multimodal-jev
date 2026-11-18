import copy
import json

import pytest
import torch

from tools.jev.audit_context_rl_training import update_targets, verify_checkpoint, audit
from tools.jev.context_rl_evaluation_plan import sha
from multimodal_jev.context_candidate_task import compile_candidates, bind_candidate_targets
from multimodal_jev.context_training_checkpoint import capture


def batch_data():
    task = dict(schema=dict(candidates=[dict(id=f'c{i}', description=f'candidate {i}') for i in range(60)]),
                thinking={'requirements':['Listen to audio.']})
    candidates = [dict(id=c['id'], definition=c['description']) for c in task['schema']['candidates']]
    pop = {str(i):dict(id=str(i), audio=f'/audio/{i}.wav', intent=f'c{i%60}') for i in range(127)}
    slots = [dict(id=str(i), weight=1., padding=False) for i in range(127)] + [dict(id='0',weight=0.,padding=True)]
    planned = dict(epoch=1, index=316, slots=slots, rank_indices=[list(range(i*32,(i+1)*32)) for i in range(4)])
    records = []
    for i,slot in enumerate(slots):
        row = pop[slot['id']]
        req = compile_candidates(row['audio'], candidates, task['thinking']['requirements'], semantics='exclusive_event')
        records.append(dict(id=row['id'], request_id=f'{row["id"]}::epoch1-batch316-slot{i}',weight=slot['weight'],
                            target=bind_candidate_targets(req, semantics='exclusive_event', exclusive_event=row['intent'])))
    return dict(batch=dict(policy_id='behavior', rank_indices=planned['rank_indices'],records=records),
                receipt=dict(behavior_policy_id='behavior', fixed_reference_policy_id='reference',update=633, queries=[{}]*128),
                planned=planned,population=pop,task=task,update=633,policy_id='behavior',reference_id='reference')


def test_tail_padding_has_unique_slot_identity_and_zero_weight():
    targets = update_targets(**batch_data())
    assert len(targets) == 128 and sum(t['real_request'] for t in targets.values()) == 127
    assert targets['0::epoch1-batch316-slot127']['weight'] == 0


def test_declared_macro_batch_binds_the_complete_rollout():
    # A wider topology passes its own global batch; the 128-row fixture must not
    # be accepted as a 384-query macro.
    args = batch_data()
    with pytest.raises(ValueError):update_targets(**args, global_batch=384)
    with pytest.raises(ValueError):update_targets(**args, global_batch=0)


@pytest.mark.parametrize('fault',['reference','behavior','rank','slot','weight','target','population','missing'])
def test_schedule_target_or_policy_mismatch_rejected(fault):
    args = batch_data()
    if fault == 'reference':args['receipt']['fixed_reference_policy_id'] = 'moving'
    elif fault == 'behavior':args['batch']['policy_id'] = 'old'
    elif fault == 'rank':args['batch']['rank_indices'] = [[0]*32]*4
    elif fault == 'slot':args['batch']['records'][-1]['request_id'] = args['batch']['records'][0]['request_id']
    elif fault == 'weight':args['batch']['records'][-1]['weight'] = 1.
    elif fault == 'target':args['batch']['records'][0]['target']['labels'][1] = True
    elif fault == 'population':args['population']['0']['intent'] = 'c2'
    elif fault == 'missing':args['receipt']['queries'].pop()
    with pytest.raises(ValueError):update_targets(**args)


@pytest.fixture
def checkpoint(tmp_path):
    contract = {'objective':'fixture'}
    full = {f'p{i}':torch.tensor([.25],dtype=torch.bfloat16) for i in range(128)}
    torch.save(full,tmp_path/'execution-0634.pt')
    for rank in range(4):
        folder = tmp_path/f'rank-{rank}';folder.mkdir()
        p = torch.nn.Parameter(torch.ones(2))
        opt = torch.optim.AdamW([p],lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False,fused=False)
        p.sum().backward();opt.step()
        opt.state[p]['step'].fill_(634)
        state = capture([('w',p)],opt,rank=rank,world_size=4,completed_updates=634,contract=contract)
        torch.save(state,folder/'checkpoint-0634.pt')
    bundle = dict(tensors={**full,'decision_token_row':torch.zeros(1,dtype=torch.bfloat16)})
    return tmp_path,634,contract,bundle


def test_actual_adam_shards_and_execution_validated(checkpoint):
    verify_checkpoint(*checkpoint)


def test_declared_width_adam_shards_accepted(tmp_path):
    # Six-rank acceptance runs reuse the same checkpoint validator; width is
    # declared by the topology, not hard-coded per configuration.
    contract = {'objective':'fixture'}
    full = {f'p{i}':torch.tensor([.25],dtype=torch.bfloat16) for i in range(128)}
    torch.save(full,tmp_path/'execution-0002.pt')
    for rank in range(6):
        folder = tmp_path/f'rank-{rank}';folder.mkdir()
        p = torch.nn.Parameter(torch.ones(2))
        opt = torch.optim.AdamW([p],lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False,fused=False)
        p.sum().backward();opt.step()
        opt.state[p]['step'].fill_(2)
        state = capture([('w',p)],opt,rank=rank,world_size=6,completed_updates=2,contract=contract)
        torch.save(state,folder/'checkpoint-0002.pt')
    bundle = dict(tensors={**full,'decision_token_row':torch.zeros(1,dtype=torch.bfloat16)})
    verify_checkpoint(tmp_path,2,contract,bundle,world_size=6)
    with pytest.raises(ValueError):verify_checkpoint(tmp_path,2,contract,bundle)


@pytest.mark.parametrize('fault',['adam_step','negative_moment','master_dtype','topology','execution'])
def test_corrupted_optimizer_or_policy_rejected(checkpoint,fault):
    root,step,contract,bundle = checkpoint
    if fault == 'execution':bundle['tensors']['p0'] = torch.zeros(1,dtype=torch.bfloat16)
    else:
        path = root/'rank-3/checkpoint-0634.pt';s = torch.load(path,weights_only=True)
        if fault == 'adam_step':s['optimizer']['state'][0]['step'].fill_(633)
        elif fault == 'negative_moment':s['optimizer']['state'][0]['exp_avg_sq'].fill_(-1)
        elif fault == 'master_dtype':s['parameters']['w'] = s['parameters']['w'].bfloat16()
        elif fault == 'topology':s['rank'] = 0
        torch.save(s,path)
    with pytest.raises(ValueError):verify_checkpoint(*checkpoint)


def test_engineering_prefix_cannot_be_a_formal_curve(tmp_path):
    c = dict(stage='task',algorithm='grpo_v1',engineering_prefix=True,training_dispatch_allowed=False)
    path = tmp_path/'config.json';path.write_text(json.dumps(c))
    image = 'sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c'
    transport = dict(state=dict(Status='exited',ExitCode=0,OOMKilled=False),container_id='fixture',image=image,
                     files={'config.json':sha(path)},status='transport_complete_independent_audit_required',
                     bad_pins=[],bad_package=[],launch_binding_mismatches=[])
    launch = dict(container_id='fixture',image=image,config_sha256=sha(path))
    with pytest.raises(ValueError,match='formal full634'):
        audit(collection=tmp_path,transport=transport,launch=launch,experiment={},schedule=None,initial_policy=None)
