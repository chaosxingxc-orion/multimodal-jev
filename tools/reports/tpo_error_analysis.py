"""Aggregate paired full-population errors; outcomes are diagnostic only."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import math
from itertools import groupby
from multiprocessing import get_context
from statistics import median
from pathlib import Path
import re

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_config import load_config
from tools.reports.heldout_error_attribution import checked
from tools.reports.decision_error_diagnosis import diagnose


def correctness_auc(observations):
    """Chance that a correct decision outscores an error, with half credit for ties."""
    errors_seen = 0
    wins = 0
    for _, tied in groupby(sorted(observations), key=lambda pair: pair[0]):
        outcomes = [correct for _, correct in tied]
        correct = sum(outcomes)
        errors = len(outcomes) - correct
        wins += correct * (errors_seen + errors / 2)
        errors_seen += errors
    comparisons = errors_seen * (len(observations) - errors_seen)
    return wins / comparisons if comparisons else None


def confidence_summary(records, outcomes):
    signals = {key: [] for key in ('candidate_median', 'top_probability', 'top_minus_second', 'top_minus_median')}
    for row in records:
        probabilities = sorted(row['prob'], reverse=True)
        background = median(probabilities)
        top = probabilities[0]
        correct = outcomes[row['id']]['top1_correct']
        for name, value in zip(signals, (background, top, top - probabilities[1], top - background), strict=True):
            signals[name].append((value, correct))
    return {name: dict(correctness_auc=correctness_auc(entries), median_across_requests=median(v for v, _ in entries))
            for name, entries in signals.items()}


def dataset(args):
    cfg, ds, comparison, metrics, seal = args
    c = cfg['inputs']
    root = Path(c['seal'])
    def sealed(rel):
        return checked(root/rel, seal['files'][rel])
    targets = sealed(f'scoring/{ds}/targets.json')
    panel = sealed(f'inference/{ds}/panel.json')
    ids = panel['candidate_ids']
    actual = [dict(id=t['id'], intent=t['gold']) for t in targets]
    fields = {q['id']: tuple(re.search(r'\b'+k+r': ([^.]+)\.', q['definition'])[1]
              for k in ('action','object','location')) for q in panel['questions']} if ds=='FSC_native31' else {}
    outcomes, report, hashes = {}, {}, {}
    for arm, pid in comparison['policy_ids'].items():
        records = []
        for shard in c['shards'][ds]:
            path = Path(c['predictions'])/shard/(arm+'.json')
            digest = file_hash(path)
            obj = checked(path,digest)
            if obj['policy_id'] != pid:
                raise ValueError('endpoint changed')
            hashes[shard+'/'+arm] = digest
            records.extend(obj['records'])
        result = diagnose(records, actual, ids)
        met = result['metrics']
        if metrics[ds][arm]['policy_id']!=pid or not all(math.isclose(met[k],v,abs_tol=1e-12,rel_tol=0)
                for k,v in metrics[ds][arm].items() if k!='policy_id'):
            raise ValueError('recomputed endpoint differs')
        outcomes[arm] = {r['id']:r for r in result['details']}
        counts, dimensions, ranks, winner = Counter(),Counter(),Counter(),Counter()
        for r in result['details']:
            winner[r['top']] += 1
            if r['top1_correct']:
                continue
            gap = r['hardest_negative_probability'] - r['gold_probability']
            counts['empty' if r['selected_count']==0 else 'multiple' if r['selected_count']>1 else 'single'] += 1
            counts['near_gap'] += gap < cfg['margin']
            counts['high_wrong'] += r['hardest_negative_probability'] >= cfg['high']
            counts['gold_tied_first'] += gap==0
            ranks[str(r['gold_rank'])] += 1
            if fields:
                diff = [k for k,a,b in zip(('action','object','location'),fields[r['gold']],fields[r['top']]) if a!=b]
                dimensions['+'.join(diff)] += 1
            if ds=='SLURP_clean_TEST60':
                dimensions['same_scenario' if r['gold'].split('_')[0]==r['top'].split('_')[0] else 'different_scenario']+=1
        report[arm] = dict(errors=sum(ranks.values()),wrong=counts,wrong_gold_competition_rank=ranks,
            mismatch=dimensions,top_confusions=result['top_confusions'],most_returned=winner.most_common(5),
            positive_nll=result['positive_nll_mean'],sum_negative_nll=result['negative_nll_sum_mean'])
        if cfg.get('confidence_diagnostics', False):
            report[arm]['confidence'] = confidence_summary(records, outcomes[arm])
    for base in ('stage1','direct'):
        transitions = Counter()
        for uid, t in outcomes['tpo'].items():
            b = outcomes[base][uid]
            key = ('retained' if t['top1_correct'] else 'regressed') if b['top1_correct'] else (
                   'fixed' if t['top1_correct'] else 'same_error' if t['top']==b['top'] else 'different_error')
            transitions[key] += 1
        report['tpo_vs_'+base] = transitions
    return ds, dict(report,prediction_sha256=hashes)


def main(cfg):
    c = cfg['inputs']
    comparison = json.loads(Path(c['comparison']).read_bytes())
    metrics = checked(c['metrics'], comparison['metrics_sha256'])
    seal = checked(Path(c['seal'])/'MANIFEST.json',comparison['seal_sha256'])
    if set(c['shards'])!=set(metrics) or seal['threshold']!=cfg['threshold'] or cfg['threshold']!=.5:
        raise ValueError('complete datasets and fixed diagnostic threshold required')
    with ProcessPoolExecutor(c['cpu_workers'],mp_context=get_context('spawn')) as pool:
        return dict(pool.map(dataset,[(cfg,ds,comparison,metrics,seal) for ds in metrics]))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config'); p.add_argument('local')
    a=p.parse_args()
    print(json.dumps(main(load_config(a.config,a.local)[0]),ensure_ascii=False,indent=2))
