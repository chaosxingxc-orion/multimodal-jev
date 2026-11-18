"""Fixed probability-error decomposition of a completed local signal report.

Uses published predictions only: no model, sampling, SSH or scheduler calls.
"""
import argparse
from collections import Counter
import fcntl
import json
import math
from pathlib import Path

from tools.reports.committed_signal_report import check_paths, read, sha, verify_sources, write


def decompose(observation, support):
    """Preserve the original TRAIN128 / Final60 / threshold .5 diagnostic."""
    ranks = observation['progress']['latest']
    step = support['actor_update']
    if (type(step) is not int or step <= 0 or support['behavior_update'] != step - 1
            or len(ranks) != 4
            or {Path(r['path']).parent.name for r in ranks} != {f'rank-{i}' for i in range(4)}):
        raise ValueError('Complete DP4 update identity required')
    for rank in ranks:
        if (rank['data']['update'] != step or rank['data']['rollout']['update'] != step - 1
                or rank['data']['rollout'] != ranks[0]['data']['rollout']):
            raise ValueError('Rank update/rollout mismatch')
    queries = ranks[0]['data']['rollout']['queries']
    if (len(queries) != 128 or len(support['queries']) != 128
            or len({q['request_id'] for q in queries}) != 128
            or len({q['id'] for q in support['queries']}) != 128):
        raise ValueError('Unique full TRAIN128 population required')
    rows = []
    for query, truth in zip(queries, support['queries'], strict=True):
        if query['request_id'].split('::epoch')[0] != truth['id']:
            raise ValueError('Query/truth identity mismatch')
        gold = truth['gold_index']
        if type(gold) is not int or not 0 <= gold < 60:
            raise ValueError('Final60 gold index required')
        row = {'id': truth['id']}
        for side in ('reference', 'behavior'):
            probs = query[side + '_record']['prob']
            if (len(probs) != 60
                    or any(not math.isfinite(p) or not 0 < p < 1 for p in probs)):
                raise ValueError('Interior finite Final60 probabilities required; endpoints are unsupported')
            selected = [i for i, p in enumerate(probs) if p >= .5]
            outcome = ('exact' if selected == [gold] else 'empty' if not selected else
                       'gold_plus_extras' if gold in selected else 'wrong_nonempty')
            positive = -math.log(probs[gold]) / 60
            negative = -sum(math.log1p(-p) for i, p in enumerate(probs) if i != gold) / 60
            nll = positive + negative
            expected = truth[side]['decision']
            f1 = 2 / (len(selected) + 1) if gold in selected else 0.
            if (not math.isclose(nll, expected['binary_nll'], rel_tol=1e-12, abs_tol=1e-14)
                    or f1 != expected['set_f1'] or len(selected) != expected['selected_count']):
                raise ValueError('Decision metrics differ from published support')
            row[side] = dict(outcome=outcome, nll=nll, positive_nll_contribution=positive,
                negative_nll_contribution=negative, gold_probability=probs[gold],
                largest_wrong_probability=max(p for i, p in enumerate(probs) if i != gold),
                gold_below_001=probs[gold] < .01,
                wrong_above_099=any(p > .99 for i, p in enumerate(probs) if i != gold),
                set_f1=f1, selected_count=len(selected))
        row['nll_delta'] = row['behavior']['nll'] - row['reference']['nll']
        rows.append(row)
    summary = {}
    for side in ('reference', 'behavior'):
        total = sum(r[side]['nll'] for r in rows)
        summary[side] = dict(mean_nll=total / 128,
            positive_nll_contribution=sum(r[side]['positive_nll_contribution'] for r in rows) / 128,
            negative_nll_contribution=sum(r[side]['negative_nll_contribution'] for r in rows) / 128,
            gold_below_001_queries=sum(r[side]['gold_below_001'] for r in rows),
            wrong_above_099_queries=sum(r[side]['wrong_above_099'] for r in rows), outcomes={})
        for outcome in ('exact', 'empty', 'gold_plus_extras', 'wrong_nonempty'):
            subset = [r[side] for r in rows if r[side]['outcome'] == outcome]
            summary[side]['outcomes'][outcome] = dict(queries=len(subset),
                mean_nll=sum(r['nll'] for r in subset) / len(subset) if subset else None,
                fraction_of_total_nll=sum(r['nll'] for r in subset) / total,
                f1_contribution_to_batch_mean=sum(r['set_f1'] for r in subset) / 128)
        if not math.isclose(summary[side]['mean_nll'], support['summaries'][side]['decision']['binary_nll']):
            raise ValueError('Population NLL differs from published support')
    return dict(queries=128, actor_update=step, behavior_update=step - 1, summary=summary,
        outcome_transitions=dict(Counter(r['reference']['outcome'] + '->' + r['behavior']['outcome'] for r in rows)),
        nll_increased_queries=sum(r['nll_delta'] > 0 for r in rows),
        nll_decreased_queries=sum(r['nll_delta'] < 0 for r in rows), details=rows,
        scope='Same authenticated TRAIN128 batch only. Outcome strata are policy-dependent; no heldout calibration, causal attribution, temporal trend or changed decision threshold.')


def published_inputs(config):
    manifest_path = Path(config['input_publication']).resolve()
    if sha(manifest_path) != config['input_publication_sha256']:
        raise ValueError('Input publication identity changed')
    manifest = read(manifest_path)
    if (manifest['schema'] != 'committed-signal-report-publication-v1'
            or manifest['complex_training_admitted'] is not False
            or set(manifest['outputs']) != {'observation.json', 'reward-support.json',
                                           'reward-groups.json', 'summary.json'}):
        raise ValueError('Complete signal publication required')
    sources = dict(manifest['sources'])
    for name, digest in manifest['outputs'].items():
        path = str(manifest_path.parent / name)
        if path in sources and sources[path] != digest:
            raise ValueError('Conflicting published identity')
        sources[path] = digest
    sources[str(manifest_path)] = config['input_publication_sha256']
    verify_sources(sources)
    return read(manifest_path.parent / 'observation.json'), read(manifest_path.parent / 'reward-support.json'), sources


def run(config_path):
    config_path = Path(config_path).resolve()
    config_hash = sha(config_path)
    config = read(config_path)
    if (set(config) != {'schema', 'input_publication', 'input_publication_sha256', 'output_directory'}
            or config['schema'] != 'probability-error-report-v1'):
        raise ValueError('Unknown configuration schema or fields')
    output = Path(config['output_directory']).resolve()
    storage = check_paths({'report': output})
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'owner.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publication = output / 'COMPLETE.json'
        if publication.exists():
            previous = read(publication)
            if previous['config_sha256'] != config_hash:
                raise ValueError('Existing publication uses another configuration')
            verify_sources(previous['sources'])
            verify_sources({str(output / name): digest for name, digest in previous['outputs'].items()})
            return dict(status='already_complete', manifest=str(publication))
        if (output / 'status.json').exists():
            raise ValueError('Incomplete run preserved; no automatic retry')
        write(output / 'status.json', dict(phase='running', config_sha256=config_hash))
        try:
            observation, support, sources = published_inputs(config)
            sources.update({str(config_path): config_hash, str(Path(__file__).resolve()): sha(__file__)})
            result = decompose(observation, support)
            result['sources'] = sources
            verify_sources(sources)
            write(output / 'probability-errors.json', result)
            write(publication, dict(schema='probability-error-report-publication-v1',
                config_sha256=config_hash, sources=sources, storage=storage,
                outputs={'probability-errors.json': sha(output / 'probability-errors.json')},
                complex_training_admitted=False))
            write(output / 'status.json', dict(phase='complete', manifest=str(publication)))
            return dict(status='complete', manifest=str(publication))
        except Exception as exc:
            write(output / 'status.json', dict(phase='failed_preserve_evidence', error=repr(exc),
                automatic_retry=False, config_sha256=config_hash))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    print(json.dumps(run(parser.parse_args().config)))


if __name__ == '__main__':
    main()
