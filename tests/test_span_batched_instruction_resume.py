import pytest

from multimodal_jev.span_batch_training_config import PROFILE
from multimodal_jev.span_batched_instruction_resume import validate_lineage


def fixture():
    previous = dict(model_path='model',base_id='base',release='data',release_sha256='d',
        schedule_sha256='s',global_batch=128,microbatch_per_rank=8,batch_order='length_grouped',
        candidate_profile=PROFILE,lora_rank=4,lora_alpha=8,lora_execution_dtype='bfloat16',lr=1e-4,
        seed=317,runtime_profile='actor_base',context_limit=4096,task_mode='direct_final',
        loss_reduction='real_request_mean',total_updates=634,updates=2)
    current = dict(previous,start_update=2,updates=2,initial_adapter='accepted',previous_root='previous')
    state = dict(updates=2,training_root='previous',adapter='accepted',candidate_profile=PROFILE)
    return previous,current,state


def test_exact_partial_and_full_continuation_allowed():
    previous,current,state=fixture()
    validate_lineage(previous,current,state)
    current['updates']=632
    validate_lineage(previous,current,state)


@pytest.mark.parametrize('field,value', [('microbatch_per_rank',16),('batch_order','original'),
    ('schedule_sha256','old'),('lr',2e-4),('seed',318),('initial_adapter','unverified'),
    ('previous_root','other'),('start_update',1),('updates',633),('updates',True),
    ('candidate_profile','old'),('total_updates',635)])
def test_changed_lineage_rejected(field,value):
    previous,current,state=fixture();current[field]=value
    with pytest.raises(ValueError): validate_lineage(previous,current,state)
