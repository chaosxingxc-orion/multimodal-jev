import hashlib
import json
from pathlib import Path

import pytest

from tools.jev.context_formal_runtime import validate, sha
from tools.jev.prepare_context_formal_runtime import transform, prepare, GUARD, REPLACEMENT
from tools.jev.prepare_context_checkpoint_runtime import transform as checkpoint_transform


def test_formal_transform_changes_only_admission(tmp_path):
    original = Path('tools/jev/train_context_online_candidate.py').read_text()
    checkpoint = checkpoint_transform(original)
    changed = transform(checkpoint)
    assert changed.replace(REPLACEMENT, GUARD) == checkpoint
    assert GUARD in original
    source = tmp_path / 'candidate.py';source.write_text(checkpoint)
    result = prepare(source=source, output=tmp_path / 'formal')
    assert result['training_dispatch_allowed'] is False
    assert (tmp_path / 'formal/train_context_formal.py').read_text() == changed
    with pytest.raises(ValueError):transform(checkpoint + '\n')


@pytest.fixture
def release(tmp_path):
    c = dict(stage='task', engineering_prefix=False, training_dispatch_allowed=True,
        start_update=0, stop_update=634, checkpoint_interval=64, global_batch=128, microbatch=8,
        physical_devices=[4,5,6,7], runtime_profile='qwen3-asr-context-bf16-packed-row-v1',
        algorithm='grpo_v1', source_policy_id='stage1-id', source_contract={'fixture':'contract'},
        source_policy=str(tmp_path / 'policy'), initial_adapter=str(tmp_path / 'adapter'),
        reference_adapter=str(tmp_path / 'adapter'))
    files = {}
    def put(name, value):
        p = tmp_path / name;p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value));files[name] = sha(p)
    for name in ['policy/MANIFEST.json', 'adapter/CONTEXT_POLICY.json',
                 'continuous.json', 'replay.json', 'checkpoint.json']:put(name, {'fixture':True})
    binding = dict(source_step=634, source_policy_id='stage1-id', reference_policy_id='stage1-id',
                   source_contract=c['source_contract'], reset_optimizer_for_task_stage=True)
    put('binding.json', binding)
    m = dict(schema='jev-context-formal-rl-release-v1', status='released',
        image='sha256:52ba94e1b6ba35f09066a92222f013ce9563fdbd83a6a5e7a60d209ac29f756c',
        files=files, acceptance=dict(continuous='continuous.json', fresh_replay='replay.json',
                                    checkpoint_frequency='checkpoint.json'), config_contract=dict(c),
        source=dict(source_step=634, reset_optimizer_for_task_stage=True, policy_id='stage1-id',
                    reference_policy_id='stage1-id', contract=c['source_contract'],
                    binding_file='binding.json', binding=binding))
    def publish():
        payload = {k:v for k,v in m.items() if k != 'release_id'}
        m['release_id'] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        path = tmp_path / 'RELEASE.json';path.write_text(json.dumps(m))
        c.update(formal_release=str(path), formal_release_sha256=sha(path), pins={str(path):sha(path)})
    publish()
    return c, m, publish


def test_runtime_identity_for_both_algorithms(release):
    c, m, publish = release
    for algorithm in ('grpo_v1', 'rloo_v1'):
        c['algorithm'] = m['config_contract']['algorithm'] = algorithm;publish()
        assert validate(c)['release_id'] == m['release_id']


@pytest.mark.parametrize('fault', ['switch_only','prefix','microbatch','source','reference','resume',
    'pending','missing_evidence','changed_asset','extra_asset','symlink','binding','config','image'])
def test_invalid_release_or_execution_rejected(release, fault):
    c, m, publish = release;root = Path(c['formal_release']).parent
    if fault == 'switch_only':c.pop('formal_release')
    elif fault == 'prefix':c['stop_update'] = 2
    elif fault == 'microbatch':c['microbatch'] = 32
    elif fault == 'source':m['source']['source_step'] = 2;publish()
    elif fault == 'reference':m['source']['reference_policy_id'] = 'moving';publish()
    elif fault == 'resume':c['resume_root'] = 'old-adam'
    elif fault == 'pending':m['status'] = 'pending';publish()
    elif fault == 'missing_evidence':m['acceptance']['fresh_replay'] = 'missing';publish()
    elif fault == 'changed_asset':(root / 'policy/MANIFEST.json').write_text('changed')
    elif fault == 'extra_asset':(root / 'policy/unbound').write_text('extra')
    elif fault == 'symlink':(root / 'policy/link').symlink_to(root / 'binding.json')
    elif fault == 'binding':m['source']['binding']['source_step'] = 2;publish()
    elif fault == 'config':c['unknown_numeric_option'] = True
    elif fault == 'image':m['image'] = 'different';publish()
    with pytest.raises((ValueError, KeyError)):validate(c)
