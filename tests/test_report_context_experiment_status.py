"""Deterministic status reporting: phases, exact process identity, reward provenance."""
import json
from pathlib import Path

import pytest

from tools.reports import report_context_experiment_status as module


def put(root, name, value):
    path = Path(root) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def stage(tmp_path, **extra):
    spec = dict(name='s', kind='k', terminal=str(tmp_path / 'finalization/terminal.json'))
    spec.update(extra)
    return spec


def test_marker_order_prefers_audit_then_launch(tmp_path):
    spec = stage(tmp_path)
    assert module.stage_phase(spec)[0] == 'absent'
    put(tmp_path / 'finalization', 'preparation-config.json', {})
    put(tmp_path / 'finalization', 'provenance.json', {})
    assert module.stage_phase(spec)[0] == 'prepared'
    put(tmp_path / 'finalization', 'config.json', {})
    assert module.stage_phase(spec)[0] == 'launching'
    put(tmp_path / 'finalization', 'audit.json', {})
    assert module.stage_phase(spec)[0] == 'audit_without_terminal'


def test_terminal_dominates_and_reports_status(tmp_path):
    spec = stage(tmp_path, accepted=['ok'])
    put(tmp_path / 'finalization', 'terminal.json', dict(exit_code=0, status='ok', updates=634))
    phase, terminal = module.stage_phase(spec)
    assert phase == 'terminal' and terminal['status'] == 'ok'
    row = module.summarize(spec)
    assert row['accepted'] is True and row['updates'] == 634


def test_progress_counts_registered_points(tmp_path):
    (tmp_path / 'root').mkdir()
    for step in (0, 32, 64):
        put(tmp_path / 'root', f'completed-{step:04d}.json', {})
    spec = dict(name='curve', kind='curve', root=str(tmp_path / 'root'),
                terminal=str(tmp_path / 'root/terminal.json'),
                progress=dict(glob=str(tmp_path / 'root/completed-*.json'), planned=11, label='选点'))
    assert module.stage_phase(spec)[0] == '3_of_11_选点'


def test_root_marker_without_terminal_path(tmp_path):
    put(tmp_path, 'retirement-complete.json', dict(retired_pids=[1], old_artifacts_preserved=True))
    spec = dict(name='async', kind='controller', root=str(tmp_path), terminal=None)
    assert module.stage_phase(spec)[0] == 'old_queue_retired'


def test_queue_events_summarize_without_guessing(tmp_path):
    assert module.queue_events(tmp_path) is None
    put(tmp_path, '000000.json', dict(event='npu_ready', tag='t', priority=0, unix=1.0))
    put(tmp_path, '000001.json', dict(event='npu_acquired', tag='t', unix=2.0))
    summary = module.queue_events(tmp_path)
    assert summary['count'] == 2
    assert summary['counts'] == dict(npu_ready=1, npu_acquired=1)
    assert summary['last']['event'] == 'npu_acquired'


def test_reused_pid_is_not_a_running_driver(tmp_path):
    record = put(tmp_path, 'started.json', dict(pid=1, argv=['x'], started_unix=0.0,
                                                process_start_ticks='not-the-real-ticks'))
    state = module.driver_state(record)
    assert state['state'] in ('running', 'pid_reused_identity_differs')
    if state['state'] == 'running':
        assert state['argv_matches'] is False
    assert module.driver_state(tmp_path / 'missing.json') == dict(state='no_driver_record')


def test_driver_record_requires_pid_and_ticks(tmp_path):
    record = put(tmp_path, 'started.json', dict(pid='1', argv=[]))
    with pytest.raises(ValueError, match='pid'):
        module.driver_state(record)


def test_audited_reward_reads_branch_means_only(tmp_path):
    assert module.audited_reward(None) is None
    assert module.audited_reward(dict(updates=[])) is None
    audit = dict(updates=[dict(update=0, real_queries=128, branches={
        'ranking': dict(mean_weighted_reward=0.5), 'existence': dict(mean_weighted_reward=0.25)})])
    rows = module.audited_reward(audit)
    assert rows == [dict(update=0, real_queries=128, ranking=0.5, existence=0.25)]


def test_summarize_prefers_independent_audit_over_provisional(tmp_path):
    put(tmp_path, 'finalization/audit.json', dict(updates=[
        dict(update=1, real_queries=128, branches={'ranking': dict(mean_weighted_reward=0.7),
                                                   'existence': dict(mean_weighted_reward=0.6)})]))
    spec = stage(tmp_path, train_reward=dict(audit=str(tmp_path / 'finalization/audit.json'),
                                            collection=str(tmp_path / 'collected'),
                                            provisional=True))
    row = module.summarize(spec)
    assert row['train_reward']['source'] == 'independent_audit'
    assert row['train_reward']['audited'] is True


def test_missing_audit_is_reported_unavailable_not_zero(tmp_path):
    put(tmp_path / 'finalization', 'config.json', {})
    spec = stage(tmp_path, train_reward=dict(audit=str(tmp_path / 'finalization/audit.json'),
                                             collection=str(tmp_path / 'nothing'),
                                             provisional=True))
    row = module.summarize(spec)
    assert row['phase'] == 'launching'
    assert row['train_reward'] == dict(source='pending', audited=False, updates=None,
                                       reason='independent audit not committed')
    line = module.reward_line(row)
    assert '不可用' in line and '0.0' not in line


def test_undispatched_stage_is_not_reported_as_unavailable_reward(tmp_path):
    spec = stage(tmp_path, train_reward=dict(audit=str(tmp_path / 'finalization/audit.json')))
    record = module.snapshot(dict(stages=[spec]))
    assert module.reward_line(record['stages'][0]) is None
    text = module.render(record)
    assert '未开始（无派发记录，尚无 reward）：s' in text
    assert '不可用' not in text


def test_non_provisional_stage_never_reads_collected_groups(tmp_path):
    spec = stage(tmp_path, train_reward=dict(collection=str(tmp_path / 'collected'),
                                             provisional=False))
    row = module.summarize(spec)
    assert row['train_reward']['source'] == 'pending'


@pytest.mark.parametrize('count,expected', [(2, '; '), (5, '平均 ranking')])
def test_reward_line_compacts_long_audits(tmp_path, count, expected):
    updates = [dict(update=i, real_queries=128, branches={'ranking': dict(mean_weighted_reward=0.5),
                                                          'existence': dict(mean_weighted_reward=0.4)})
               for i in range(count)]
    put(tmp_path, 'finalization/audit.json', dict(updates=updates))
    spec = stage(tmp_path, train_reward=dict(audit=str(tmp_path / 'finalization/audit.json')))
    line = module.reward_line(module.summarize(spec))
    assert expected in line


def test_gate_reports_declared_predecessor_status(tmp_path):
    predecessor = put(tmp_path, 'predecessor/terminal.json', dict(exit_code=0, status='done'))
    spec = stage(tmp_path)
    put(tmp_path / 'finalization', 'waiting.json', dict(predecessor=dict(terminal=str(predecessor))))
    assert module.gate(spec) == dict(path=str(predecessor), present=True, status='done')
    spec_flat = stage(tmp_path, terminal=str(tmp_path / 'other/terminal.json'))
    put(tmp_path / 'other', 'waiting.json', dict(terminal=str(predecessor)))
    assert module.gate(spec_flat)['status'] == 'done'


def test_dev_point_keeps_registered_metrics_only(tmp_path):
    put(tmp_path, 'audit.json', dict(source_step=128, status='complete',
                                     result=dict(metrics=dict(query_macro_set_f1=0.4,
                                                              top1_event_accuracy=0.5,
                                                              confusion=[]))))
    assert module.dev_point(tmp_path) == dict(step=128, status='complete',
                                              metrics=dict(query_macro_set_f1=0.4,
                                                           top1_event_accuracy=0.5))
    assert module.dev_point(tmp_path / 'missing') is None


def test_ladder_completion_ignores_retired_stages(tmp_path):
    done = stage(tmp_path, accepted=['ok'])
    put(tmp_path / 'finalization', 'terminal.json', dict(exit_code=0, status='ok'))
    retired = dict(name='old', kind='old', terminal=str(tmp_path / 'gone/terminal.json'),
                   retired='superseded')
    record = module.snapshot(dict(stages=[done, retired]))
    assert record['ladder_complete'] is True


def test_render_states_dev_unavailable_instead_of_fabricating(tmp_path):
    spec = stage(tmp_path, dev_points=str(tmp_path / 'points/finalization'),
                 dev_points_planned=11)
    record = module.snapshot(dict(stages=[spec]))
    text = module.render(record)
    assert '0/11' in text and '未产生' in text
    assert '无 SSH/Docker/模型调用/派发/重启' in text


def test_provisional_reward_matches_audit_statistic(tmp_path):
    torch = pytest.importorskip('torch')
    groups = []
    for real in (True, False):
        groups.append(dict(real_request=real, dual_credit=dict(branches={
            'ranking': dict(weight=torch.tensor([1., 1.]), rewards=torch.ones(2, 16)),
            'existence': dict(weight=torch.tensor([1., 1.]), rewards=torch.full((2, 16), .5))})))
    collection = tmp_path / 'run'
    collection.mkdir()
    torch.save(groups, collection / 'groups-0007.pt')
    rows = module.provisional_reward(collection)
    assert rows[0]['update'] == 7 and rows[0]['real_queries'] == 1
    assert rows[0]['ranking'] == pytest.approx(2.0) and rows[0]['existence'] == pytest.approx(1.0)
