import pytest
from test_context_infonce_margin import O


def test_legacy_checkpoint_cannot_restore_under_new_loss_contract():
    from test_context_infonce_training_audit import fixture,restore_shard
    c,state=fixture();c['instruction_objective']=O
    with pytest.raises(ValueError):restore_shard(state,c,rank=0,step=1)
