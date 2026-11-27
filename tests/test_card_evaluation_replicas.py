from pathlib import Path

import pytest

from multimodal_jev import evaluation_blocks as blocks
from multimodal_jev.baseline import file_hash
from multimodal_jev.context_execution import content_key
from multimodal_jev.local_checkpoint import commit_json
from tools.reports import card_evaluation_replicas as replicas
from tools.reports.screen_card_evaluation_replicas import panel, telemetry


def test_replay_reports_changed_probabilities_and_does_not_touch_archive(tmp_path, monkeypatch):
    monkeypatch.setattr('multimodal_jev.local_checkpoint.check_paths', lambda _: None)
    archive = tmp_path/'archive'
    archive.mkdir()
    block, = blocks.ordered_blocks([dict(dataset='d', arm='a', partition=0, offset=0,
                                          ids=['recording'], widths=[2], work=10)], ['a'])
    plan = dict(schema=blocks.SCHEMA, namespace='baseline', policies={'a':'policy'},
                choice=False, blocks=[block])
    manifest = dict(plan=plan, sha256=content_key(plan))
    commit_json(archive/'plan.json', manifest)
    frozen = tmp_path/'frozen.json'
    commit_json(frozen, manifest)

    def output(probability):
        return {'prob':dict(policy_id='policy', records=[
            dict(id='recording', prob=[0.5, probability])])}

    blocks.publish(archive, manifest, block, output(0.75))
    archived_digest = file_hash(blocks.artifact(archive, block))

    def run_worker(spec):
        root = Path(spec['queue']['root'])
        blocks.publish(root, manifest, block, output(0.76))
        return dict(blocks=[{'block':block['key']}])

    monkeypatch.setattr(replicas, 'worker', run_worker)
    result = replicas.run(dict(queue={'root':str(tmp_path/'replay')},
        result_path=str(tmp_path/'result.json'),
        replica_replay=dict(plan=str(frozen), sha256=file_hash(frozen), archive=str(archive))))
    assert result['probability_differences'] == 1
    assert result['pairs'] == 2
    assert file_hash(blocks.artifact(archive, block)) == archived_digest


def test_panel_covers_both_policies_and_preserves_actual_host_measurement():
    plan = {'blocks':[dict(dataset='d', arm=arm, key=arm+str(index))
                      for arm in ('old', 'new') for index in range(5)]}
    assert [b['key'] for b in panel(plan, {'d':3})['blocks']] == [
        'old0','old2','old4','new0','new2','new4']
    with pytest.raises(ValueError, match='quota'):
        panel(plan, {'unrelated':3})
    raw = '''
| 4     910B4-1             | OK            | 155.4                43                      0    / 0                |
| 0                         | 0000:01:00.0  | 73                   0    / 0                49152/ 65536            |
| 7     910B4-1             | OK            | 110.4                43                      0    / 0                |
| 0                         | 0000:42:00.0  | 21                   0    / 0                16384/ 65536            |
'''
    assert telemetry(raw) == {4:{'aicore':73,'hbm':75.}, 7:{'aicore':21,'hbm':25.}}
