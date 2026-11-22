"""CPU-only paired error analysis of an audited, sample-block probability evaluation."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json

import numpy as np

from multimodal_jev.evaluation_blocks import checked_plan, read_block
from multimodal_jev.context_execution import map_io
from tools.reports.diagnose_probability_projection import gap, rank_summary, read, recording_evidence


def diagnose(root, score_pin, config_pin, plan_pin, metadata, minds):
    spec = read(root / 'entry/score.json', config_pin)['macros']['NATIVE_SEAL_CONTAINER_CONFIG']
    metrics = read(root / 'metrics.json', score_pin)
    queue = root / Path(spec['queue']['root']).name
    manifest = checked_plan(queue, spec['queue']['namespace'])
    if manifest['sha256'] != plan_pin:
        raise ValueError('unexpected block plan')
    plan = manifest['plan']
    rows = defaultdict(dict)

    def visit(b):
        return b, read_block(queue, manifest, b)['prob']
    def append(index, payload, pid):
        if payload['policy_id'] != pid:
            raise ValueError('different policy')
        for row in payload['records']:
            if row['id'] in index:
                raise ValueError('duplicate row')
            index[row['id']] = row['prob']
    publications, _ = map_io(visit, plan['blocks'], workers=spec['io_workers'])
    for b, payload in publications:
        append(rows[b['dataset'], b['arm']], payload, metrics['datasets'][b['dataset']]['arms'][b['arm']]['policy_id'])

    panels = {}
    for ds, receipt in metrics['datasets'].items():
        cfg = spec['endpoints'][ds]['cache']
        seal = Path(cfg['seal']['root'])
        m = read(seal / 'MANIFEST.json', metrics['seals'][ds])
        def sealed(rel):
            return read(seal / rel, m['files'][rel])
        truth = sealed(f'scoring/{ds}/targets.json')
        rel = f'inference/{ds}/panel.json'
        if rel in m['files']:
            ids = sealed(rel)['candidate_ids']
        else:
            options = sealed(f'inference/{ds}/panels.json')
            layouts = {tuple(q['id'] for q in v['questions']) for v in options.values()}
            if len(layouts) != 1:
                raise ValueError('candidate order differs across questions')
            ids = list(layouts.pop())
        roster = [t['id'] for t in truth]
        y = np.array([ids.index(t['gold']) for t in truth])
        mats, winners = {}, {}
        for arm, scores in receipt['arms'].items():
            ix = rows[ds, arm]
            for part in spec['reused_predictions'][ds].get(arm, {}).get('shards', []):
                append(ix, read(Path(part['path']), part['sha256']), scores['policy_id'])
            assert len(roster) == len(ix) == scores['queries'] and set(roster) == set(ix)
            p = np.array([ix[u] for u in roster])
            assert p.shape == (len(y), len(ids)) and np.isfinite(p).all()
            win = p.argmax(1)
            assert np.isclose((win == y).mean(), scores['top1_accuracy'], atol=1e-12, rtol=0)
            assert np.isclose(p.max(1).mean(), scores['mean_top1_probability'], atol=1e-12, rtol=0)
            mats[arm], winners[arm] = p, win
        p, q = mats['tpo'], mats['baseline']
        win, old = winners['tpo'], winners['baseline']
        bad, prev = win != y, old != y
        reg, fix, persistent = bad & ~prev, ~bad & prev, bad & prev
        assert (int(reg.sum()), int(fix.sum())) == tuple(receipt['paired']['baseline'][k] for k in ('regressed','fixed'))
        mg, pg, ng = gap(p,y)
        oldmg, oldpg, oldng = gap(q,y)
        def pairs(mask, w):
            c = Counter((ids[a], ids[b]) for a,b in zip(y[mask], w[mask]))
            return [[a,b,n] for (a,b),n in sorted(c.items(), key=lambda z:(-z[1], z[0]))]
        def median(x,mask):
            return float(np.median(x[mask])) if mask.any() else None
        comp = np.arange(len(y))
        delta_neg = p[comp,win]-q[comp,win]
        delta_gold = pg-oldpg
        def hist(mask,w):
            return dict(sorted(Counter(ids[i] for i in w[mask]).items()))
        result = dict(n=len(y), wrong=rank_summary(p,y,bad), persistent=rank_summary(p,y,persistent),
                      regressed=rank_summary(p,y,reg), fixed=int(fix.sum()),
                      stage1_correct_regressed=int((reg & (winners['stage1']==y)).sum()),
                      wrong_pairs=pairs(bad,win), regression_pairs=pairs(reg,win), repaired_pairs=pairs(fix,old),
                      fixed_rival_on_regression=dict(gold_down=int((reg & (delta_gold<0)).sum()),
                          winner_up=int((reg & (delta_neg>0)).sum()),
                          both=int((reg & (delta_gold<0) & (delta_neg>0)).sum()),
                          median_gold_before=median(oldpg,reg),median_gold_after=median(pg,reg),
                          median_wrong_before=median(q[comp,win],reg),median_wrong_after=median(ng,reg),
                          median_margin_before=median(oldmg,reg),median_margin_after=median(mg,reg)),
                      gold=hist(np.ones(len(y),dtype=bool),y),
                      returned={arm:hist(np.ones(len(y),dtype=bool),w) for arm,w in winners.items()})
        ev = recording_evidence(ds, truth, p, y, ids, bad, metadata / 'test.jsonl', minds)
        if ds.startswith('SLURP'):
            # Classify only regressions, allowing all other recordings as peers.
            groups = defaultdict(list)
            for i, t in enumerate(truth):
                groups[t['group'], t['gold']].append(i)
            peers = Counter()
            for i in np.flatnonzero(reg):
                item = truth[i]
                other = [j for j in groups[item['group'],item['gold']]
                         if truth[j]['audio_sha256'] != item['audio_sha256']]
                peers['correct_peer' if any(not bad[j] for j in other)
                      else 'all_peers_wrong' if other else 'no_peer'] += 1
            ev['regression_peers'] = dict(peers)
        result['recording_evidence'] = ev
        panels[ds] = result
    return dict(score_sha256=score_pin, config_sha256=config_pin, plan_sha256=plan_pin, panels=panels)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ('root','score_pin','config_pin','plan_pin','metadata','minds'):
        cli.add_argument(arg, type=Path if arg in ('root','metadata','minds') else str)
    print(json.dumps(diagnose(**vars(cli.parse_args())), ensure_ascii=False, indent=2, allow_nan=False))
