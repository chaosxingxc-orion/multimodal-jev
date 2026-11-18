"""Repeated CPU sampling at archived probabilities; never a training job."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import torch
from tools.reports.analyze_context_counterfactual_credit import exact_direction
from tools.reports.replay_context_reward_groups import run as authenticate


def estimate(p, gold, actions):
    """R x G x N actions -> two R x N raw reward-ascent logit estimates."""
    if (p.ndim != 1 or not torch.isfinite(p).all() or ((p <= 0) | (p >= 1)).any()
            or type(gold) is not int or not 0 <= gold < len(p)
            or actions.ndim != 3 or actions.shape[-1] != len(p)
            or actions.shape[0] < 1 or actions.shape[1] < 2
            or not ((actions == 0) | (actions == 1)).all()):
        raise ValueError('interior probabilities and complete singleton binary groups required')
    a = actions.to(p.dtype)
    counts = a.sum(-1)
    rewards = 2*a[..., gold]/(counts+1)
    g = a.shape[1]
    centered = (rewards-rewards.mean(1, keepdim=True))*(g/(g-1))
    rloo = (centered[..., None]*(a-p)).mean(1)
    others = counts[..., None]-a
    delta = -2*a[..., gold, None]/((others+2)*(others+1))
    delta[..., gold] = 2/(others[..., gold]+2)
    leg = (p*(1-p)*delta).mean(1)
    return {'rloo': rloo, 'local_expectation': leg}


def moments(estimates, exact):
    mean = estimates.mean(0)
    error = (estimates-exact).square().sum(-1)
    return dict(mse=float(error.mean()),
                mse_standard_error=float(error.std(correction=1)/len(error)**.5),
                variance_trace_population=float((estimates-mean).square().sum(-1).mean()),
                empirical_mean_error_squared=float((mean-exact).square().sum()),
                zero_fraction=float((estimates == 0).all(-1).double().mean()),
                exact_squared_norm=float(exact.square().sum()))


def run(observation, release, schedule, receipts, cache_proof, *, repeats=512, seed=170041):
    if repeats < 2 or seed < 0: raise ValueError('at least two trials and nonnegative seed required')
    started = time.monotonic()
    bound = authenticate(observation, release, schedule, receipts, cache_proof)
    queries = json.loads(Path(observation).read_text())['progress']['latest'][0]['data']['rollout']['queries']
    rows = []
    for index, (query, truth) in enumerate(zip(queries, bound['queries'], strict=True)):
        if query['request_id'] != truth['request_id']: raise ValueError('order mismatch')
        p = torch.tensor(query['behavior_record']['prob'], dtype=torch.float64)
        gold = truth['gold_index']
        expected_reward, exact = exact_direction(p, gold)
        actions = torch.rand((repeats, 16, len(p)), dtype=torch.float64,
                             generator=torch.Generator().manual_seed(seed+index)) < p
        estimates = estimate(p, gold, actions)
        rows.append(dict(request_id=truth['request_id'], source_group=truth['source_group'],
            threshold_exact_set=truth['threshold_exact_set'], seed=seed+index,
            expected_F1=expected_reward, gold_probability=float(p[gold]),
            methods={name: moments(value, exact) for name, value in estimates.items()}))
    strata = {}
    for label, subset in [('all', rows), ('threshold_correct', [r for r in rows if r['threshold_exact_set']]),
                          ('threshold_wrong', [r for r in rows if not r['threshold_exact_set']])]:
        methods = {name: {k: sum(r['methods'][name][k] for r in subset)/len(subset)
                         for k in subset[0]['methods'][name] if k != 'mse_standard_error'}
                   for name in ('rloo', 'local_expectation')}
        strata[label] = dict(queries=len(subset), methods=methods,
            queries_with_lower_LEG_mse=sum(r['methods']['local_expectation']['mse'] < r['methods']['rloo']['mse'] for r in subset))
    sources = dict(bound['sources'])
    for path in (Path(__file__), Path('tools/reports/analyze_context_counterfactual_credit.py')):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(schema='context-credit-repeated-sampling-v1', repeats=repeats, group_size=16,
        random_dtype='float64_CPU; newly drawn diagnostic samples, not archived action replay',
        seed_base=seed, torch_version=torch.__version__, actor_update=bound['actor_update'],
        sources=sources, seconds=time.monotonic()-started, strata=strata, queries=rows,
        limits=['Fixed previously selected TRAIN128 snapshot, not population/generalization evidence.',
                'Raw expected reward/output-logit derivatives; not actor gradients or GRPO normalized objective.',
                'MSE, finite-sample empirical variance and empirical mean error are distinct; finite mean error is not estimator bias.',
                'RLOO and LEG use paired G16 samples; reward computation budgets differ; no speed comparison.',
                'New FP64 random draws are mathematical diagnostics, not formal FP32 sampler parity.',
                'No optimizer, NPU, heldout evaluation, new dispatch or runtime mutation.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('observation', 'release', 'schedule', 'receipts', 'cache-proof', 'output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--repeats', type=int, default=512)
    parser.add_argument('--seed', type=int, default=170041)
    args = vars(parser.parse_args()); output = args.pop('output')
    torch.set_num_threads(1)
    result = run(**args)
    with Path(output).open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({'strata': result['strata'], 'seconds': result['seconds']}, indent=2))
