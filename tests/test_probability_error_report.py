from copy import deepcopy
import json
import math

import pytest

from tools.reports import probability_error_report as report


def fixture():
    queries, truths = [], []
    for i in range(128):
        p = [.1] * 60
        if i % 4 in (0, 2):
            p[0] = .8
        if i % 4 in (2, 3):
            p[1] = .9
        chosen = [j for j, v in enumerate(p) if v >= .5]
        decision = dict(binary_nll=-(math.log(p[0]) + sum(math.log1p(-v) for v in p[1:])) / 60,
                        set_f1=2 / (len(chosen) + 1) if 0 in chosen else 0.,
                        selected_count=len(chosen))
        queries.append(dict(request_id=f'q{i}::epoch0', reference_record={'prob': p},
                            behavior_record={'prob': p.copy()}))
        truths.append(dict(id=f'q{i}', gold_index=0, reference={'decision': decision},
                           behavior={'decision': decision.copy()}))
    ranks = [dict(path=f'run/rank-{i}/update-0064.json',
                  data=dict(update=64, rollout=dict(update=63, queries=deepcopy(queries)))) for i in range(4)]
    mean = sum(t['reference']['decision']['binary_nll'] for t in truths) / 128
    support = dict(actor_update=64, behavior_update=63, queries=truths,
                   summaries={side: {'decision': {'binary_nll': mean}} for side in ('reference', 'behavior')})
    return {'progress': {'latest': ranks}}, support


def test_four_outcomes_and_nll_decomposition():
    observation, support = fixture()
    result = report.decompose(observation, support)
    assert result['actor_update'] == 64
    assert result['nll_increased_queries'] == result['nll_decreased_queries'] == 0
    assert set(result['outcome_transitions'].values()) == {32}
    for side in ('reference', 'behavior'):
        summary = result['summary'][side]
        assert all(v['queries'] == 32 for v in summary['outcomes'].values())
        assert math.isclose(sum(v['fraction_of_total_nll'] for v in summary['outcomes'].values()), 1)
        assert math.isclose(summary['mean_nll'], summary['positive_nll_contribution'] + summary['negative_nll_contribution'])


@pytest.mark.parametrize('value', [0., 1., float('nan'), float('inf')])
def test_invalid_probabilities_not_clipped(value):
    observation, support = fixture()
    for rank in observation['progress']['latest']:
        rank['data']['rollout']['queries'][0]['behavior_record']['prob'][0] = value
    with pytest.raises(ValueError):
        report.decompose(observation, support)


@pytest.mark.parametrize('mutation', ['population', 'gold', 'identity', 'rank', 'metric'])
def test_population_binding_rejections(mutation):
    observation, support = fixture()
    if mutation == 'population':
        support['queries'].pop()
    elif mutation == 'gold':
        support['queries'][0]['gold_index'] = -1
    elif mutation == 'identity':
        support['queries'][0]['id'] = 'unknown'
    elif mutation == 'rank':
        observation['progress']['latest'][3]['data']['update'] = 65
    else:
        support['queries'][0]['behavior']['decision']['set_f1'] = 0
    with pytest.raises(ValueError):
        report.decompose(observation, support)


def test_publication_idempotence_and_changed_input_rejected(tmp_path, monkeypatch):
    observation, support = fixture()
    upstream = tmp_path / 'upstream'
    upstream.mkdir()
    for name, value in [('observation.json', observation), ('reward-support.json', support),
                        ('reward-groups.json', {}), ('summary.json', {})]:
        (upstream / name).write_text(json.dumps(value))
    manifest = upstream / 'COMPLETE.json'
    manifest.write_text(json.dumps(dict(schema='committed-signal-report-publication-v1',
        complex_training_admitted=False, sources={},
        outputs={p.name: report.sha(p) for p in upstream.iterdir()})))
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(dict(schema='probability-error-report-v1',
        input_publication=str(manifest), input_publication_sha256=report.sha(manifest),
        output_directory=str(tmp_path / 'output'))))
    monkeypatch.setattr(report, 'check_paths', lambda *a: [])
    assert report.run(config)['status'] == 'complete'
    monkeypatch.setattr(report, 'decompose', lambda *a: pytest.fail('recomputed'))
    assert report.run(config)['status'] == 'already_complete'
    (upstream / 'observation.json').write_text('{}')
    with pytest.raises(ValueError, match='Source changed'):
        report.run(config)
