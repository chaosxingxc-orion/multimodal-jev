import json
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


def test_full_utilization_counts_idle_cards_and_rejects_missing_time(tmp_path):
    from tools.reports.verify_card_evaluation_throughput import utilization

    telemetry_path = tmp_path/'telemetry.jsonl'
    with telemetry_path.open('w') as log:
        for stamp, core, hbm in [(0, 60, 70), (10, 100, 90), (20, 60, 70)]:
            log.write(json.dumps(dict(unix=stamp, devices={
                '0': dict(aicore=core, hbm=hbm), '1': dict(aicore=0, hbm=20),
            }))+'\n')
    actual = utilization(telemetry_path, (5, 15), [0, 1], 10)
    assert actual['by_device']['0'] == {'aicore':90., 'hbm':85.}
    assert (actual['aicore'], actual['hbm']) == (45., 52.5)
    with pytest.raises(ValueError, match='gap'):
        utilization(telemetry_path, (5, 15), [0, 1], 9)
    with pytest.raises(ValueError, match='cover'):
        utilization(telemetry_path, (5, 21), [0, 1], 10)


def test_formal_timing_only_counts_recomputed_blocks(tmp_path):
    from tools.reports.verify_card_evaluation_throughput import timings

    package = tmp_path/'package'
    entry = package/'entry'
    ops = tmp_path/'ops'
    entry.mkdir(parents=True)
    ops.mkdir()

    def write(path, value):
        path.write_text(json.dumps(value))

    block_plan = entry/'block-plan.json'
    reuse = dict(arm='stage1', plan={'path':'/sealed/plan.json', 'sha256':'pinned'})
    write(block_plan, dict(macros={'NATIVE_SEAL_CONTAINER_CONFIG': dict(queue={'reuse_policy':reuse})}))
    worker = 'block-worker-0'
    deploy = dict(chains=[[[dict(name=worker, devices=[0])]]], host='181', image='image', tag='eval',
                  entry_pins={block_plan.name:file_hash(block_plan)})
    write(package/'operations-local.json', dict(macros={'NATIVE_SEAL_OPERATIONS_CONFIG':deploy}))
    write(ops/'terminal.json', dict(exit_code=0, status='declared_stages_finished'))
    write(ops/'block-plan-collected.json', dict(blocks=2, reused_blocks=1, plan_sha256='current'))
    write(ops/(worker+'-launch.json'), dict(container_id='worker'))
    write(ops/(worker+'-inspect.json'), dict(Id='worker', Image='image', Name='/eval-'+worker,
          State=dict(Status='exited', ExitCode=0, StartedAt='1970-01-01T00:00:05Z',
                     FinishedAt='1970-01-01T00:00:20Z')))
    receipt = dict(devices=[0], plan_sha256='current', replicas_per_device=6,
                   blocks=[dict(block='b', dataset='test', arm='tpo', recordings=1,
                                claimed_unix=7, first_input_unix=9, committed_unix=18)])
    receipt_path = ops/(worker+'-collected.json')
    write(receipt_path, receipt)
    plan = dict(blocks=[dict(key='a', dataset='test', arm='stage1', ids=['recording']),
                        dict(key='b', dataset='test', arm='tpo', ids=['recording'])])
    result = timings(tmp_path, plan, 6, reused_policy=reuse)
    assert (result['scoring_start'], result['scoring_end'], result['worker_wall_seconds']) == (9, 18, 15)
    with pytest.raises(ValueError, match='completed-policy reuse'):
        timings(tmp_path, plan, 6, reused_policy=dict(reuse, plan={'sha256':'another-run'}))
    receipt['blocks'][0].update(block='a', arm='stage1')
    write(receipt_path, receipt)
    with pytest.raises(ValueError, match='foreign'):
        timings(tmp_path, plan, 6, reused_policy=reuse)
