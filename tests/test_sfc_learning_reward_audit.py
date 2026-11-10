import json
from pathlib import Path
import sys
import pytest

from multimodal_jev.baseline import atomic_json,file_hash
from multimodal_jev.sfc_jev_shards import make_plan
from test_sfc_jev import load_tool
from test_sfc_jev_merge import fixture as merge_fixture

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/jev'))


def test_complete_learning_only_controls(tmp_path,monkeypatch):
    _,path,_,release,_=merge_fixture(tmp_path,monkeypatch,'learning')
    module=load_tool('audit_sfc_learning_rewards')
    # No TEST labels exist; the estimator has no reason to access them.
    for p in release.glob('test*.targets.jsonl'):p.unlink()
    result=module.audit(path,file_hash(path))
    assert result['counts']==dict(bundles=2,decisions=12,functions=4,active_functions=4,inactive_functions=0,no_call_bundles=0)
    assert result['source_groups']==1 and result['model_called'] is False
    assert result['synthetic_controls']['oracle']['all_decisions_correct']==2
    assert result['synthetic_controls']['oracle']['reward_fraction_of_oracle']==1
    assert result['synthetic_controls']['consistent_empty']['total_reward']==0
    assert result['synthetic_controls']['constant_false']['total_reward']==0  # Correct absent offsets malformed calls.
    assert result['synthetic_controls']['malformed']['total_reward']==-12


def test_selection_labels_not_parsed_or_scored(tmp_path,monkeypatch):
    _,path,_,release,_=merge_fixture(tmp_path,monkeypatch,'learning')
    module=load_tool('audit_sfc_learning_rewards');plan=json.loads(path.read_text())
    wanted={uid for s in plan['shards'] for uid in s['ordered_bundle_ids']}
    target=release/'train_id.targets.jsonl';values=target.read_text().splitlines(keepends=True)
    target.write_text(''.join(line if json.loads(line)['id'] in wanted else 'UNPARSEABLE SELECTION LINE\n' for line in values))
    manifest=json.loads((release/'MANIFEST.json').read_text());manifest['splits']['train_id']['targets_sha256']=file_hash(target)
    atomic_json(release/'MANIFEST.json',manifest)
    partition_path=Path(plan['partition']['path']);partition=json.loads(partition_path.read_text())
    partition['manifest_sha256']=file_hash(release/'MANIFEST.json');atomic_json(partition_path,partition)
    plan=make_plan(release,'train_id',12,partition_path=partition_path,partition_hash=file_hash(partition_path),part='learning')
    atomic_json(path,plan)
    assert module.audit(path,file_hash(path))['counts']['bundles']==2


@pytest.mark.parametrize('part',[None,'selection'])
def test_nonlearning_population_rejected_before_target_read(tmp_path,monkeypatch,part):
    _,path,_,release,_=merge_fixture(tmp_path,monkeypatch,part)
    for p in release.glob('*.targets.jsonl'):p.unlink()
    with pytest.raises(ValueError,match='learning population'):
        load_tool('audit_sfc_learning_rewards').audit(path,file_hash(path))


def test_target_drift_rejected(tmp_path,monkeypatch):
    _,path,_,release,_=merge_fixture(tmp_path,monkeypatch,'learning')
    (release/'train_id.targets.jsonl').write_text('changed')
    with pytest.raises(ValueError,match='target file changed'):
        load_tool('audit_sfc_learning_rewards').audit(path,file_hash(path))
