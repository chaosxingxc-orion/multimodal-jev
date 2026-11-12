import json
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.span_pair_pilot import pilot_rows
from test_span_transfer_runtime import fixture


@pytest.mark.parametrize('fault',[None,'weight','source','pin','initial','runtime','steps'])
def test_one_step_protocol_cannot_change_old_study_or_initializer(tmp_path,fault):
    study,batch=fixture(tmp_path)
    batch.update(base_id='base')
    source=tmp_path/'step.json';source.write_text(json.dumps(batch))
    # The fixture has an initializer adapter; bind its base for this engineering probe.
    protocol=dict(schema='jev-pair-engineering-pilot-v1',algorithm='exact-complement-pair-marginal-ppo-v1',
        pair_weights=[0.,.5],num_updates=1,auxiliary_weight=0.,clip_ratio=.2,optimizer_start='fresh',lr=1e-4,
        seed=17,arm='structured',batch_index=0,source_step=str(source),initial_adapter_id='initial',
        base_id='base',runtime_admission='runtime',pins={})
    batch['runtime_admission']='runtime';source.write_text(json.dumps(batch))
    protocol['pins'][str(source)]=file_hash(source)
    if fault=='initial':protocol['initial_adapter_id']='different'
    elif fault=='runtime':protocol['runtime_admission']='other'
    elif fault=='steps':protocol['num_updates']=12
    path=tmp_path/'pilot.json';path.write_text(json.dumps(protocol))
    config=dict(protocol=str(path),pair_weight=1. if fault=='weight' else .5,pins={str(path):file_hash(path)})
    if fault=='pin':config['pins']={}
    elif fault=='source':source.write_text('{}')
    if fault:
        with pytest.raises(ValueError):pilot_rows(config)
    else:
        _,_,rows,pairs=pilot_rows(config)
        assert len(rows)==8 and pairs.shape==(4,2)
