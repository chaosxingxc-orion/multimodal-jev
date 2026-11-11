import json
from pathlib import Path
import pytest
from multimodal_jev.baseline import file_hash
from multimodal_jev.policy_identity import checkpoint_identity
from multimodal_jev.rl_run import load_rollout_run
from multimodal_jev.task_schema import JevExample,DecisionSchema


def fixture(tmp_path,split='train'):
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir()
    (checkpoint/'model.safetensors').write_bytes(b'fixture weights, not a real model')
    (checkpoint/'config.json').write_text('{}')
    policy,files=checkpoint_identity(checkpoint)
    audio=tmp_path/'audio.wav';audio.write_bytes(b'fixture audio')
    ex=JevExample('ex',split,str(audio),DecisionSchema('s','noul','Question?'),True,
                  {'audio_sha256':file_hash(audio)})
    manifest=tmp_path/'manifest';manifest.write_text(json.dumps(ex.to_record())+'\n')
    run=tmp_path/'run';run.mkdir()
    config={'checkpoint':str(checkpoint),'manifest':str(manifest),'example_ids':['ex'],
            'sampling':{'temperature':1.,'top_p':1.,'top_k':-1,'n':2,'max_tokens':16}}
    identity={'config':config,'policy_id':policy,'checkpoint_files':files,'manifest_sha256':file_hash(manifest)}
    (run/'identity.json').write_text(json.dumps(identity))
    records=[{'example_id':'ex','replicate':i,'policy_id':policy,'token_ids':[1,2],
              'log_probs':[-1.,-2.],'text':'fixture','finish_reason':'stop'} for i in range(2)]
    (run/'rollouts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (run/'terminal.json').write_text(json.dumps({'exit_code':0}))
    (run/'result.json').write_text(json.dumps({'status':'passed','policy_id':policy,'prompts':1,'rollouts':2,
                                              'rollouts_sha256':file_hash(run/'rollouts.jsonl')}))
    return run,checkpoint


def test_rollout_receipt_identity_and_train_only(tmp_path):
    run,_=fixture(tmp_path)
    identity,examples,records=load_rollout_run(run)
    assert len(examples)==1 and len(records)==2
    (run/'rollouts.jsonl').write_text('changed')
    with pytest.raises(ValueError,match='responses changed'):load_rollout_run(run)


def test_processor_changes_invalidate_policy_identity(tmp_path):
    run,checkpoint=fixture(tmp_path)
    (checkpoint/'config.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='checkpoint identity'):load_rollout_run(run)


def test_nontraining_and_failed_sampling_rejected(tmp_path):
    run,_=fixture(tmp_path,'valid')
    with pytest.raises(ValueError,match='TRAIN'):load_rollout_run(run)
    (run/'terminal.json').write_text('{"exit_code":1}')
    with pytest.raises(ValueError,match='successful rollout'):load_rollout_run(run)


def test_invalid_response_logprob_rejected():
    from types import SimpleNamespace as NS
    from multimodal_jev.rl_update import validate_rollouts
    from multimodal_jev.rollout import Rollout
    records=[Rollout('ex',i,'policy',(1,),(-1.,),'text','stop') for i in range(2)]
    records[0]=Rollout('ex',0,'policy',(1,),(float('nan'),),'text','stop')
    with pytest.raises(ValueError,match='logprobs'):validate_rollouts([NS(example_id='ex')],records,'policy',2)


def test_diagnostic_rollout_cannot_train(tmp_path):
    from multimodal_jev.rl_run import train
    run, _ = fixture(tmp_path)
    identity = json.loads((run/'identity.json').read_text())
    identity['config']['diagnostic_only'] = True
    (run/'identity.json').write_text(json.dumps(identity))
    with pytest.raises(ValueError, match='diagnostic-only'):
        load_rollout_run(run)
    assert len(load_rollout_run(run, allow_diagnostic=True)[2]) == 2
    with pytest.raises(ValueError, match='diagnostic-only'):
        train({'rollout_run':str(run)}, tmp_path/'must-not-create')
    assert not (tmp_path/'must-not-create').exists()
