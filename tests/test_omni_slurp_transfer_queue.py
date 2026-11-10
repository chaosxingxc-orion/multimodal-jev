import json
from pathlib import Path
import sys
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.consumption import canonical

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))
import run_omni_slurp_transfer_queue as module


def test_queue_refuses_replay_and_missing_roster_before_dispatch(tmp_path,monkeypatch):
    plan=tmp_path/'plan.json';plan.write_bytes(canonical({'shards':[{'id':'s0'},{'id':'s1'}]}))
    protocol=tmp_path/'transfer-protocol.json';protocol.write_bytes(canonical(dict(shard_plan=str(plan),prompts={'parent':'a','p1':'b'})))
    validation=tmp_path/'train-validation.json';validation.write_text('{}')
    intent=dict(protocol_sha256=file_hash(protocol),validation_sha256=file_hash(validation),validation_root='/validation',
        roster=[dict(shard=s,prompt=p) for s in ['s0','s1'] for p in ['parent','p1']])
    path=tmp_path/'queue-intent.json';path.write_bytes(canonical(intent))
    monkeypatch.setattr(module,'verify',lambda _: {})
    monkeypatch.setattr(module,'dispatch',lambda _:pytest.fail('must not dispatch'))
    (tmp_path/'p1/s1').mkdir(parents=True)
    with pytest.raises(FileExistsError,match='prior preparation'):module.run(tmp_path)
    assert not (tmp_path/'dispatch-intent.json').exists()
    intent['roster'].pop();path.write_bytes(canonical(intent))
    with pytest.raises(ValueError,match='whole registered'):module.run(tmp_path)


def test_prepare_rejects_missing_train_validation_before_creating_queue(tmp_path,monkeypatch):
    def failed(_):raise ValueError('TRAIN not complete')
    monkeypatch.setattr(module,'verify',failed)
    root=tmp_path/'queue'
    with pytest.raises(ValueError,match='TRAIN'):module.prepare(root,tmp_path/'protocol',tmp_path/'train',tmp_path/'native',[6,7])
    assert not root.exists()
