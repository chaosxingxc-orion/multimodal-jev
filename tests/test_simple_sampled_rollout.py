import copy
import hashlib
import json
import math
import pytest
torch=pytest.importorskip('torch')
from multimodal_jev.simple_sampled_rollout import sample_record, fixed_reference, actor_item, actor_batch


def record(policy='base:current'):
    identity=dict(policy_id=policy,request={'audio':'wave'},inputs={'input_ids':'hash'},positions=[[5]],
                  candidate_ids=[list(range(60))],sampling=dict(temperature=1.0,policy='independent_finite_categorical'))
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',', ':')).encode()).hexdigest()
    return dict(request_id=digest,request_identity=identity,candidate_logprobs=[[-math.log(60)]*60])


def sample(r,**kwargs):
    options=dict(expected_request_id=r['request_id'],expected_identity=r['request_identity'],
                 policy_id='base:current',seed=317,update=0,slot=0)
    options.update(kwargs)
    return sample_record(r,**options)


def test_reproducible_label_free_sampling_does_not_mutate_global_rng():
    r=record();state=torch.random.get_rng_state();a=sample(r);b=sample(r)
    assert torch.equal(state,torch.random.get_rng_state())
    assert torch.equal(a['actions'],b['actions']) and a['actions'].shape==(1,16,1)
    assert torch.equal(a['old_joint'],a['behavior'][0].gather(1,a['actions'][:,:,0]))
    assert not a['old_joint'].requires_grad and len(torch.unique(a['actions']))>1
    assert sample(r,slot=1)['receipt']['sampling_sha256']!=a['receipt']['sampling_sha256']
    assert sample(r,update=1)['receipt']['sampling_sha256']!=a['receipt']['sampling_sha256']


@pytest.mark.parametrize('change',[dict(policy_id='base:stale'),dict(expected_request_id='wrong'),
    dict(expected_identity={}),dict(seed=True),dict(update=-1),dict(slot=-1),dict(group_size=8)])
def test_reject_stale_or_unbound_sampling(change):
    with pytest.raises(ValueError):sample(record(),**change)


def test_reject_bad_distribution_without_sampling_or_filtering():
    r=record();r['candidate_logprobs'][0][0]=float('nan')
    with pytest.raises(ValueError):sample(r)
    r=record();r['candidate_logprobs']=[[0.]*60]
    with pytest.raises(ValueError):sample(r)


def test_fixed_reference_binds_identical_input_to_different_policy():
    current=record();ref=record('base:stage1')
    p=fixed_reference(ref,expected_identity=current['request_identity'],reference_policy_id='base:stage1')
    assert p[0].shape==(1,60) and not p[0].requires_grad
    with pytest.raises(ValueError):fixed_reference(current,expected_identity=current['request_identity'],reference_policy_id='base:stage1')
    bad=copy.deepcopy(current['request_identity']);bad['inputs']['input_ids']='other-wave'
    with pytest.raises(ValueError):fixed_reference(ref,expected_identity=bad,reference_policy_id='base:stage1')


def actor_fixture():
    from multimodal_jev.span_request_identity import tensor_identity
    inputs=dict(input_ids=torch.tensor([[1,2,3]]),attention_mask=torch.ones(1,3,dtype=torch.long),
                input_features=torch.zeros(1,1,3),input_features_mask=torch.ones(1,1,dtype=torch.long))
    item=dict(inputs=inputs,positions=torch.tensor([[2]]),candidates=[list(range(60))],policy_id='base:current',
              targets=torch.tensor([[3]]),supervised=torch.ones(1,1,dtype=torch.bool),span_weights=torch.ones(1,1),
              request_weight=1.,real_request=True)
    records=[]
    for policy in ['base:current','base:stage1']:
        r=record(policy);identity=r['request_identity'];identity.update(inputs={k:tensor_identity(v) for k,v in sorted(inputs.items())},positions=[[2]])
        r['request_id']=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',', ':')).encode()).hexdigest();records.append(r)
    return item,records


def test_actor_bridge_samples_independently_of_labels_and_separates_padding_slots():
    item,(b,r)=actor_fixture();kw=dict(expected_identity=b['request_identity'],reference_policy_id='base:stage1',seed=317,update=0,slot=0)
    a,receipt=actor_item(item,b,r,**kw)
    other=dict(item,targets=torch.tensor([[48]]));c,_=actor_item(other,b,r,**kw)
    assert torch.equal(a['actions'],c['actions']) and a['request_id']==receipt['actor_request_id']
    padding=dict(item,request_weight=0.,real_request=False)
    p,_=actor_item(padding,b,r,**dict(kw,slot=1))
    assert not p['real_request'] and p['request_weight']==0 and p['request_id']!=a['request_id']


@pytest.mark.parametrize('fault',['audio','support','position','policy','target','mask','weight'])
def test_actor_bridge_rejects_cross_input_or_supervision_mismatch(fault):
    item,(b,r)=actor_fixture()
    if fault=='audio':item['inputs']['input_features'].fill_(1)
    elif fault=='support':item['candidates'][0].reverse()
    elif fault=='position':item['positions'].fill_(1)
    elif fault=='policy':item['policy_id']='base:stale'
    elif fault=='target':item['targets'].fill_(60)
    elif fault=='mask':item['supervised'].fill_(False)
    else:item['span_weights'].fill_(.5)
    with pytest.raises(ValueError):actor_item(item,b,r,expected_identity=b['request_identity'],reference_policy_id='base:stage1',seed=317,update=0,slot=0)


def batch_fixture():
    item,(b,r)=actor_fixture()
    entries=[dict(identity=copy.deepcopy(b['request_identity']),item=copy.deepcopy(item)) for _ in range(4)]
    entries[-1]['item'].update(real_request=False,request_weight=0.)
    results=[dict(passed=True,backend='vllm',base_id='base',adapter_id=policy,records=[row])
             for policy,row in [('current',b),('stage1',r)]]
    kwargs=dict(policy_id='base:current',reference_policy_id='base:stage1',seed=317,update=0)
    return entries,results,kwargs


def test_complete_batch_keeps_padding_and_uses_unique_backend_requests():
    entries,results,kw=batch_fixture()
    items,receipt=actor_batch(entries,*results,**kw)
    assert len(items)==4 and receipt['global_real_requests']==3 and receipt['unique_behavior_requests']==1
    assert len({i['request_id'] for i in items})==4
    assert all(i['actions'].shape==(1,16,1) for i in items)


@pytest.mark.parametrize('fault',['missing','duplicate','extra','hf','stale','version','partial_batch','padding','all_padding'])
def test_batch_rejects_incomplete_or_foreign_transport(fault):
    entries,results,kw=batch_fixture()
    if fault=='missing':results[0]['records']=[]
    elif fault=='duplicate':results[0]['records']*=2
    elif fault=='extra':results[0]['records'].append(dict(results[0]['records'][0],request_id='foreign'))
    elif fault=='hf':results[0]['backend']='hf'
    elif fault=='stale':results[0]['adapter_id']='old'
    elif fault=='version':entries[0]['item']['policy_id']='base:old'
    elif fault=='partial_batch':entries.pop()
    elif fault=='padding':entries[-1]['item']['request_weight']=1.
    else:
        for e in entries:e['item'].update(real_request=False,request_weight=0.)
    with pytest.raises(ValueError):actor_batch(entries,*results,**kw)
