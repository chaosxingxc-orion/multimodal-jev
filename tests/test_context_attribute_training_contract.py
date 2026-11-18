import copy
import json
from pathlib import Path

import pytest
import torch

from multimodal_jev.context_training_checkpoint import capture, restore
from tools.reports.context_attribute_actor_candidate import generate
from tools.reports.context_attribute_training_contract import regroup, checkpoint_contract, prepare_schedule, load_attribute_schedule
from multimodal_jev.baseline import file_hash

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'.local/decision-objective-research'


def fixtures():
    menu=json.loads((BASE/'slurp-native-attribute-prob-candidate-v1/catalog.json').read_text())
    config=json.loads((BASE/'context-rl-grpo-arithmetic-v2-package/config.json').read_text())
    return menu,config


def manifest(menu,config,arm):
    _,actor=generate(menu,arm,source_root=ROOT)
    return dict(schema='context-attribute-training-schedule-v1', actor_contract_id=actor['contract_id'],
        catalog_sha256=actor['catalog_sha256'], release_sha256=config['release_sha256'],
        input_arm='direct' if arm=='direct' else 'q_input',recordings=40513,updates=634,
        global_batch=128,world_size=4,microbatch=8,accumulation=4,
        source_schedule_sha256='a'*64,processor_records_sha256='b'*64,
        processor_transport_sha256='c'*64,query_order_sha256='d'*64,
        lengths_sha256='e'*64,batches_sha256='f'*64,batches_file_sha256='0'*64,
        training_dispatch_allowed=False)


def test_full_real_schedule_regroup_preserves_exposure(tmp_path):
    # Full real two-epoch order; lengths are deliberately synthetic and reversed.
    inputs=[json.loads(s) for s in (BASE/'slurp-native-attribute-prob-candidate-v1/inputs.jsonl').read_text().splitlines()]
    original=[]
    for epoch in range(2):
        original.extend(json.loads((BASE/f'context-rl-grpo-arithmetic-v2-package/source/schedule/epoch-{epoch}.json').read_text())['batches'])
    def records():
        for i,asset in enumerate(inputs):
            yield dict(id=asset['id'],audio_sha256=asset['audio_sha256'],records=[
                dict(arm=arm,tokens=offset+255-i%256,tensors={'input_features':{'shape':[1,128,100+i%128]}})
                for arm,offset in [('direct',2000),('q_input',3000)]])
    derived,identity=regroup(original,records(),inputs,arm='q_input')
    credit,other=regroup(original,records(),inputs,arm='q_credit')
    assert derived==credit and identity==other and len(derived)==634
    assert any(a['rank_indices']!=b['rank_indices'] for a,b in zip(original,derived))
    for before,after in zip(original,derived):
        assert {k:v for k,v in before.items() if k!='rank_indices'}=={k:v for k,v in after.items() if k!='rank_indices'}
        assert sorted(x for row in after['rank_indices'] for x in row)==list(range(128))
        assert [len(row) for row in after['rank_indices']]==[32]*4
    # Independent order reconstruction for the first global batch.
    positions={r['id']:i for i,r in enumerate(inputs)}
    expected=sorted(range(128),key=lambda j:(3000+255-positions[original[0]['slots'][j]['id']]%256,
                                           100+positions[original[0]['slots'][j]['id']]%128,j))
    ranks=derived[0]['rank_indices']
    recovered=[]
    for step in range(4):
        for rank in (range(4) if step%2==0 else range(3,-1,-1)):
            recovered.extend(ranks[rank][step*8:(step+1)*8])
    assert recovered==expected
    menu,c=fixtures();m=manifest(menu,c,'q_input');m.update(identity)
    original_root=BASE/'context-rl-grpo-arithmetic-v2-package/source/schedule'
    m['source_schedule_sha256']=file_hash(original_root/'MANIFEST.json')
    (tmp_path/'batches.json').write_text(json.dumps(derived))
    m['batches_file_sha256']=file_hash(tmp_path/'batches.json')
    (tmp_path/'MANIFEST.json').write_text(json.dumps(m))
    kwargs=dict(original_schedule=original_root,population={r['id']:r for r in inputs},menu=menu,arm='q_input',source_root=ROOT)
    loaded,_=load_attribute_schedule(tmp_path,file_hash(tmp_path/'MANIFEST.json'),**kwargs)
    assert loaded==derived
    with pytest.raises(ValueError):load_attribute_schedule(tmp_path,'0'*64,**kwargs)
    # Even a newly hashed candidate cannot change the frozen query order.
    derived[0]['slots'][0]['weight']=0.
    from tools.reports.context_attribute_actor_candidate import digest
    m['batches_sha256']=digest(derived)
    (tmp_path/'batches.json').write_text(json.dumps(derived))
    m['batches_file_sha256']=file_hash(tmp_path/'batches.json')
    (tmp_path/'MANIFEST.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):load_attribute_schedule(tmp_path,file_hash(tmp_path/'MANIFEST.json'),**kwargs)


def test_checkpoint_identity_enforced_by_existing_restore():
    menu,c=fixtures()
    contracts={arm:checkpoint_contract(c,menu,arm,manifest(menu,c,arm),source_root=ROOT)
               for arm in ('direct','q_input','q_credit')}
    p=torch.nn.Parameter(torch.tensor([.5,-.2],dtype=torch.float32))
    opt=torch.optim.AdamW([p],lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False,fused=False)
    p.square().sum().backward();opt.step();opt.zero_grad()
    state=capture([('head',p)],opt,rank=0,world_size=4,completed_updates=1,contract=contracts['q_input'])
    def fresh():
        q=torch.nn.Parameter(torch.ones(2));o=torch.optim.AdamW([q],lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.,foreach=False,fused=False)
        return q,o
    for bad in [contracts['direct'],contracts['q_credit']]:
        q,o=fresh()
        with pytest.raises(ValueError):restore([('head',q)],o,state,rank=0,world_size=4,completed_updates=1,contract=bad)
        assert torch.equal(q,torch.ones(2)) and not o.state
    for field in ['algorithm','schedule','attribute','objective']:
        bad=copy.deepcopy(contracts['q_input']);bad[field]='changed'
        q,o=fresh()
        with pytest.raises(ValueError):restore([('head',q)],o,state,rank=0,world_size=4,completed_updates=1,contract=bad)
        assert not o.state
    q,o=fresh();restore([('head',q)],o,state,rank=0,world_size=4,completed_updates=1,contract=contracts['q_input'])
    # A subsequent real CPU Adam update must match uninterrupted execution.
    for param,optimizer in [(p,opt),(q,o)]:
        (param*torch.tensor([2.,-3.])).sum().backward();optimizer.step()
    assert torch.equal(p,q)
    for k,v in opt.state[p].items():assert torch.equal(v,o.state[q][k])


@pytest.mark.parametrize('fault',['reference','algorithm','arm','population','release'])
def test_mismatched_configuration_rejected(fault):
    menu,c=fixtures();m=manifest(menu,c,'q_credit')
    if fault=='reference':c['source_policy_id']='bad'
    if fault=='algorithm':c['algorithm']='ppo'
    if fault=='arm':m['input_arm']='direct'
    if fault=='population':m['recordings']=2
    if fault=='release':m['release_sha256']='bad'
    with pytest.raises(ValueError):checkpoint_contract(c,menu,'q_credit',m,source_root=ROOT)


def test_no_schedule_publication_without_terminal_audit(tmp_path):
    with pytest.raises(FileNotFoundError):prepare_schedule(source_root=ROOT,
        original_schedule=tmp_path/'old',original_sha256='a'*64,collection=tmp_path/'collection',
        package=tmp_path/'package',transport=tmp_path/'missing',anchors=tmp_path/'anchors',
        arm='q_input',output=tmp_path/'new')
    assert not (tmp_path/'new').exists()


@pytest.mark.parametrize('fault',['missing','duplicate','audio','shape','length'])
def test_bad_processor_evidence_rejected(fault):
    assets=[dict(id='a',audio_sha256='a'*64)]
    record=dict(id='a',audio_sha256='a'*64,records=[
        dict(arm=arm,tokens=3000,tensors={'input_features':{'shape':[1,128,100]}})
        for arm in ('direct','q_input')])
    rows=[record]
    if fault=='missing':rows=[]
    if fault=='duplicate':rows.append(copy.deepcopy(record))
    if fault=='audio':record['audio_sha256']='b'*64
    if fault=='shape':record['records'][1]['tensors']['input_features']['shape']=[128,100]
    if fault=='length':record['records'][1]['tokens']=5000
    with pytest.raises(ValueError):regroup([],rows,assets,arm='q_credit')
