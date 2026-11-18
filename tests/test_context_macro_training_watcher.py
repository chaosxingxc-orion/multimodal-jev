import pytest
from tools.reports.watch_context_macro_training import deadline


@pytest.mark.parametrize('status,running', [('created', False), ('paused', False),
    ('restarting', False), ('running', True), ('exited', True)])
def test_nonterminal_states_preserve_job(status, running):
    r = dict(status='observed_exact_attempt', containers={'client': {'state':
        {'Status': status, 'Running': running}}}, next_ordinary_observation_unix=1900)
    assert deadline(r, 1000) == 1900


@pytest.mark.parametrize('status', ['exited', 'dead'])
def test_only_confirmed_client_terminal_stops_watcher(status):
    r = dict(status='observed_exact_attempt', containers={'client': {'state':
        {'Status': status, 'Running': False}}})
    assert deadline(r, 1000) is None


def test_timeout_is_not_terminal_and_consumes_interval():
    assert deadline(dict(status='observation_failed_preserve_handles'), 1000) == 1900
    assert deadline(dict(status='not_due_no_remote_call',
        next_ordinary_observation_unix=1500), 1000) == 1500
