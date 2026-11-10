import json
from pathlib import Path

import pytest

from test_sfc_jev import load_tool
from multimodal_jev.baseline import file_hash


def setup(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'tools/jev'))
    tool=load_tool('sample_sfc_omni_learning')
    base=tmp_path/'base.json';base.write_text('{}')
    part=tmp_path/'partition.json';part.write_text(json.dumps({'learning':['g']}))
    row={'id':'one','source_group':'g'}
    (tmp_path/'train_id.requests.jsonl').write_text(json.dumps(row)+'\n')
    monkeypatch.setattr(tool,'select_inputs',lambda _:([row],{'release':str(tmp_path)},{'check':True}))
    monkeypatch.setattr(tool,'validate_partition',lambda rows,partition: {r['id']:r for r in rows})
    cfg=dict(baseline_config=str(base),baseline_config_sha256=file_hash(base),
             partition=str(part),partition_sha256=file_hash(part),group_size=2,seed=17,max_tokens=1024)
    return tool,cfg,part


def test_sampling_input_binding_is_separate_from_greedy_protocol(tmp_path,monkeypatch):
    tool,cfg,_=setup(tmp_path,monkeypatch)
    assert tool.load_inputs(cfg)[1][0]['id']=='one'
    assert not (tmp_path/'train_id.targets.jsonl').exists()


@pytest.mark.parametrize('change',['base','partition','selection','replicates','seed','budget','extra'])
def test_sampling_rejects_drift_before_loading_engine(tmp_path,monkeypatch,change):
    tool,cfg,part=setup(tmp_path,monkeypatch)
    if change=='base':cfg['baseline_config_sha256']='0'*64
    if change=='partition':cfg['partition_sha256']='0'*64
    if change=='selection':
        part.write_text(json.dumps({'learning':['other']}));cfg['partition_sha256']=file_hash(part)
    if change=='replicates':cfg['group_size']=1
    if change=='seed':cfg['seed']=18
    if change=='budget':cfg['max_tokens']=128
    if change=='extra':cfg['adapter']='new'
    with pytest.raises(ValueError):tool.load_inputs(cfg)
