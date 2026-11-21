import pytest
from tests.test_context_infonce_training_audit import fixture, restore_shard
from tools.reports.audit_context_infonce_training import expected_contract
from tools.reports.context_infonce_trainer import generate
from pathlib import Path


def test_changed_layout_rejects_old_optimizer_checkpoint():
    c,state=fixture()
    c['prompt_layout']='jev-stable-prefix-audio-readout-v1'
    assert expected_contract(c)['prompt_layout']==c['prompt_layout']
    with pytest.raises(ValueError):restore_shard(state,c,rank=0,step=1)
    source=generate(Path(__file__).resolve().parents[1])
    assert "contract['prompt_layout']=c['prompt_layout']" in source
