"""Paired complete-population SFC prompt comparison using unchanged strict scores.

This pure layer does not authenticate containers. Callers must independently
accept every shard and freeze the candidate roster before providing outputs.
"""
from collections import Counter, defaultdict

from .sfc_jev_score import score_bundle

RANK_FIELDS = ('complete_conversations', 'all_decisions_correct', 'native_calls_exact')
GUARD_FIELDS = ('all_decisions_correct', 'native_calls_exact', 'format_valid', 'schema_valid')
TIE_ORDER = ('parent', 'p1', 'p2')


def population_scores(requests, targets, predictions):
    """Score every declared decision; gold never affects which decisions run."""
    if (not requests or len({r['id'] for r in requests}) != len(requests)
            or len({t['id'] for t in targets}) != len(targets)
            or {r['id'] for r in requests} != {t['id'] for t in targets}
            or any(r['split'] != 'train_id' for r in requests)):
        raise ValueError('unique aligned TRAIN population required')
    gold = {t['id']: t for t in targets}
    by_id = {r['id']: [] for r in requests}
    groups = defaultdict(list)
    for row in requests:
        groups[row['source_group']].append(row)
    for group, rows in groups.items():
        turns = sorted(len(r['audio_sha256']) for r in rows)
        if turns != ([1,2] if group.startswith('level_3') else [1]):
            raise ValueError('complete original conversations required')
    for p in predictions:
        if p['example_id'] not in by_id:
            raise ValueError('unknown prediction bundle')
        by_id[p['example_id']].append(p)
    totals = Counter()
    bundles = {}
    activity = {k: Counter(functions=0, correct=0) for k in ('active', 'inactive')}
    for row in requests:
        uid = row['id']
        result = score_bundle(row, gold[uid], by_id[uid])
        totals.update(result['counts'])
        totals.update(bundles=1, all_decisions_correct=int(result['all_decisions_correct']),
                      native_calls_exact=int(result['native_calls_exact']))
        for d in row['input']['decisions']:
            if d['id'].endswith('.calls'):
                bucket = 'active' if gold[uid]['answers'][d['id']] else 'inactive'
                prefix = d['id'].rsplit('.',1)[0]
                correct = all(result['decisions'][prefix+'.'+s]['correct'] for s in ('present','absent','calls'))
                activity[bucket].update(functions=1, correct=int(correct))
        bundles[uid] = {k: result[k] for k in ('all_decisions_correct', 'native_calls_exact')}
    conversations = {g: all(bundles[r['id']]['all_decisions_correct'] for r in rs)
                     for g, rs in groups.items()}
    totals.update(conversations=len(groups), complete_conversations=sum(conversations.values()))
    return dict(counts=dict(totals), bundles=bundles, conversations=conversations,
                activity={k:dict(v) for k,v in activity.items()})


def compare(requests, targets, predictions_by_prompt, roster):
    """Return a deterministic selection decision, not independent TEST evidence.

    roster is registered before selection inference. Missing/failed candidates
    must block selection, never be silently dropped based on partial results.
    """
    if (type(roster) is not list or not 2 <= len(roster) <= 3 or roster[0] != 'parent'
            or roster != [p for p in TIE_ORDER if p in roster]
            or set(predictions_by_prompt) != set(roster)):
        raise ValueError('complete preregistered parent plus candidate roster required')
    scores = {p:population_scores(requests, targets, predictions_by_prompt[p]) for p in roster}
    parent = scores['parent']; eligible = ['parent']; reasons = {}
    for p in roster[1:]:
        counts = scores[p]['counts']
        failed = [k for k in GUARD_FIELDS if counts[k] < parent['counts'][k]]
        reasons[p] = failed
        if not failed:
            eligible.append(p)
    # Stable order makes exact ties prefer parent, then p1, then p2.
    winner = max(eligible, key=lambda p:tuple(scores[p]['counts'][k] for k in RANK_FIELDS))
    paired = {}
    for p in roster[1:]:
        transitions = {}
        for level, fields in (('conversations', ('complete_conversations',)),
                              ('bundles', ('all_decisions_correct','native_calls_exact'))):
            for field in fields:
                before = parent[level]
                after = scores[p][level]
                if set(before) != set(after):raise ValueError('paired identities differ')
                pairs = [(before[k],after[k]) if level=='conversations' else
                         (before[k][field],after[k][field]) for k in before]
                transitions[field] = dict(fixed=sum(not a and b for a,b in pairs),
                    regressed=sum(a and not b for a,b in pairs),
                    both_correct=sum(a and b for a,b in pairs),
                    both_incorrect=sum(not a and not b for a,b in pairs))
        paired[p] = transitions
    return dict(schema='sfc-jev-prompt-selection-decision-v1',roster=roster,winner=winner,
        ranking=list(RANK_FIELDS),nonregression_guards=list(GUARD_FIELDS),
        ineligible_reasons=reasons,paired=paired,
        scores={p:{k:v for k,v in score.items() if k in ('counts','activity')} for p,score in scores.items()},
        scope='Complete supplied population, strict paired prompt selection only. Caller must verify '
              'selection partition, frozen roster and actual containers. No TEST confirmation, '
              'statistical significance, unseen-task generalization or interactive execution claim.')
