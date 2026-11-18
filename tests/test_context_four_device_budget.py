import json
from pathlib import Path

import pytest

from multimodal_jev.context_topology_batch_plan import derive
from tools.reports.context_infonce_trainer import topology
from tools.reports.context_infonce_formal import formal_config
from tools.reports.audit_context_infonce_training import checkpoint_steps
from tools.reports.context_infonce_tail import contract as tail_contract

ROOT=Path(__file__).resolve().parents[1]


def design():
    return json.loads((ROOT/'configs/experiments/context-two-stage-v0.0.11.json').read_text())


def stage1_config():
    from test_context_stage_lora import contract
    c=design()['stage1']
    return dict(c,physical_devices=[4,5,6,7],microbatch=8,source_contract=contract(c['lora']),
        source_root='/old/source',schedule='/old/source/schedule',parent_schedule='/old/source/parent-schedule',pins={})


def test_both_stages_use_exactly_four_devices_with_full_source_exposures():
    d=design()
    for name,total,batch,per_epoch,pad in [('stage1',318,256,159,191),('stage2',634,128,317,63)]:
        s=d[name];c=s['batch_config'];p=derive(c)
        assert c['training_devices']+c['inference_instances']==d['total_device_budget']==4
        assert p['global_batch']==s['global_batch']==batch
        assert p['total_updates']==s['total_updates']==total
        assert p['updates_per_epoch']==s['updates_per_epoch']==per_epoch
        assert p['tail_real_queries']==65 and p['tail_padding_slots']==pad
        assert c['source_queries']*c['epochs']==81026
    assert d['stage2']['batch_config']['inference_instances']==2
    assert d['stage2']['macro_queries']==128 and d['stage2']['rollout_queries']==32


def test_dp4_formal_endpoint_curve_and_actual_tail_indices():
    c=stage1_config();d=design()
    assert topology(c,4,['davinci4','davinci5','davinci6','davinci7'])['local_batch']==64
    formal=formal_config(c,'/old','/new',d)
    assert formal['checkpoint_interval']==32 and formal['reference_endpoint']==318
    assert checkpoint_steps(formal)==[32,64,96,128,160,192,224,256,288,318]
    assert formal['stage_lora_transition']['reuse_stage1_adapter'] is True
    tail=tail_contract(dict(c,experiment_kind='infonce_tail_admission',start_update=0,stop_update=2,
        resume_root=None,engineering_prefix=True,training_dispatch_allowed=False,checkpoint_interval=1))
    assert tail['schedule_indices']==[158,159] and tail['real_queries']==[65,256]
    with pytest.raises(ValueError):topology(c,8,['davinci'+str(i) for i in range(8)])


def test_formal_launch_refuses_historical_eight_card_package(tmp_path):
    from tools.reports.context_infonce_formal import launch
    (tmp_path/'config.json').write_text(json.dumps({'batch_config':{'training_devices':8,'inference_instances':0}}))
    with pytest.raises(ValueError,match='four training devices'):
        launch(dict(package=str(tmp_path),formal_training_dispatch_allowed=True),{})
