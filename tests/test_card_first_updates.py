import copy

import pytest

from tools.reports.collect_card_first_updates import summarize


def test_triplet_startup_requires_global_loss_and_matching_optimizer_steps():
    config = dict(physical_devices=[0, 1], global_batch=4, instruction_objective=dict(
        kind='intent_triplet_infonce_v1', probability_margin=dict(gap_weight=1., anchor_weight=.25)))
    metrics = {'jev/infonce': .5, 'jev/gap': .1, 'jev/anchor': .4, 'actor/pg_loss': .7, 'actor/grad_norm': 1.}
    row = dict(real_queries=4, update=1, learning_rate=.0001, metrics=metrics, preparation_seconds=.01, total_seconds=1.,
               peak_allocated=123, optimizer_audit={'head': dict(numel=1, state='initialized', step=1)})
    rows = [row, copy.deepcopy(row)]
    assert summarize(rows, 1, config)['loss'] == .7
    rows[1]['metrics']['actor/pg_loss'] = .6
    with pytest.raises(ValueError, match='across ranks'):
        summarize(rows, 1, config)
    rows[0]['metrics']['actor/pg_loss'] = .6
    with pytest.raises(ValueError, match='components'):
        summarize(rows, 1, config)
    rows[1]['optimizer_audit']['head']['step'] = 0
    with pytest.raises(ValueError, match='optimizer update missing'):
        summarize(rows, 1, config)
