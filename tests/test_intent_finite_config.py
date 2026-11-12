from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/jev'))
from train_intent_finite_npu import validate_config


def config():return dict(arm='P',global_requests=8,step=0,lora_execution_dtype='bfloat16',lr=1e-4,seed=317)


def test_no_silent_optimizer_restart_or_precision_switch():
    validate_config(config())
    for key,value in [('step',1),('step',True),('previous_root','old'),('initial_optimizer_checkpoint','old'),
                      ('lora_execution_dtype','float32'),('global_requests',7),('lr',float('nan')),('seed',True),('arm','SFT')]:
        cfg=config();cfg[key]=value
        with pytest.raises(ValueError):validate_config(cfg)
