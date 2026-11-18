from tools.reports.watch_context_existing_jobs import next_due


def test_not_due_preserves_existing_deadline():
    assert next_due(dict(status='not_due_no_remote_call',
        next_ordinary_observation_unix=1800), 950) == 1800


def test_terminal_stops_only_after_successful_observation():
    assert next_due(dict(status='observed_original_handle',
        state=dict(Running=False)), 950) is None
    assert next_due(dict(status='observation_failed_preserve_original_handle',
        next_ordinary_observation_unix=1800), 950) == 1800


def test_unexpected_error_defers_full_interval():
    assert next_due(dict(status='watcher_observation_error'), 950) == 1850
