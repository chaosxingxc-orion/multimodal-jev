"""Separate model errors, matched audio evidence and probability diagnostics."""
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import math

from .context_reference_dataset import digest


def validate_options(options):
    fields = {'mode', 'seed', 'easy_fraction', 'replay_class_priority', 'evidence'}
    if options['evidence'].get('schema')=='jev-paired-train-evidence-v3':
        fields.update(('min_unique_recordings','probability_examples_per_type'))
        if type(options.get('min_unique_recordings')) is not int or options['min_unique_recordings'] < 1:
            raise ValueError('positive minimum unique TRAIN recording budget required')
        if type(options.get('probability_examples_per_type')) is not int or options['probability_examples_per_type'] < 0:
            raise ValueError('nonnegative probability sampling budget required')
    if set(options) != fields:
        raise ValueError('explicit paired TRAIN sampling policy required')
    if (options['mode'] != 'paired_train_v1' or type(options['seed']) is not int or options['seed'] < 0
            or type(options['replay_class_priority']) is not int or options['replay_class_priority'] < 1
            or type(options['easy_fraction']) not in (int, float) or not 0 < options['easy_fraction'] < 1):
        raise ValueError('invalid replay policy')


def validate_binding(evidence, rows, release_sha256):
    if (evidence['schema'] not in {'jev-paired-train-evidence-v1','jev-paired-train-evidence-v2','jev-paired-train-evidence-v3'}
            or evidence['release_sha256'] != release_sha256
            or evidence['rows_sha256'] != digest(sorted(rows, key=lambda r: r['id']))):
        raise ValueError('matched evidence must bind the same full clean TRAIN release')


def observe(job):
    uid, intent, gold, probs, candidates = job
    if (len(probs) != len(candidates)
            or any(type(p) not in (int, float) or not 0 < p < 1 for p in probs)):
        raise ValueError('complete interior native probabilities required')
    positive = probs[gold]
    highest = max(probs)
    wrong = [p for i, p in enumerate(probs) if i != gold]
    return dict(id=uid, intent=intent,
        rank='wrong' if positive < highest else 'tied' if positive == max(wrong) else 'correct',
        gold_rank_strict=1+sum(p > positive for p in probs),
        competing_top_ids=[c for i, c in enumerate(candidates) if i != gold and probs[i] == highest],
        positive_nll=-math.log(positive), negative_nll=-sum(math.log1p(-p) for p in wrong)/len(wrong),
        positive_brier=(1-positive)**2, negative_brier=sum(p*p for p in wrong)/len(wrong))


def sample(rows, candidate_ids, probabilities, options, *, execution=None, timings=None):
    evidence = options['evidence']
    by_id = {r['id']: r for r in rows}
    label_index = {label: i for i, label in enumerate(candidate_ids)}
    if (len(by_id) != len(rows) or set(by_id) != set(probabilities) or not rows
            or len(label_index) != len(candidate_ids) or len(label_index) < 2
            or len({r['audio_sha256'] for r in rows}) != len(rows)):
        raise ValueError('unique clean audio and complete native reference required')
    for row in rows:
        if (row['role'] != 'learning' or row['official_split'] != 'train'
                or row['known_pending_source'] is not False or row['intent'] not in label_index):
            raise ValueError('only admitted clean TRAIN can supply evidence or labels')
    jobs = [(uid, r['intent'], label_index[r['intent']], probabilities[uid], candidate_ids)
            for uid, r in sorted(by_id.items())]
    if execution is None:
        diagnostics = list(map(observe, jobs))
    else:
        from .context_execution import map_cpu
        diagnostics, stats = map_cpu(observe, jobs, workers=execution['cpu_workers'],
            min_parallel=execution['min_parallel'], chunk=execution['chunk'], window=execution['window'])
        if timings is not None:
            timings['classify'] = {k:v for k,v in stats.items() if k != 'task_seconds'}
    observed = {r['id']: r for r in diagnostics}
    eligible = set(evidence['eligible_ids'])
    if len(eligible) != len(evidence['eligible_ids']) or not eligible <= by_id.keys():
        raise ValueError('invalid isolated TRAIN population')
    hard, stable = defaultdict(dict), defaultdict(dict)
    pair_counts = Counter()

    def compare(kind, left, right):
        a, b = observed[left], observed[right]
        if (left == right or (a['intent'] == b['intent']) != (kind != 'cue_flip')
                or left not in eligible or right not in eligible):
            raise ValueError('matched evidence does not match permitted training targets')
        pair_counts[kind] += 1
        for x, y in ((a, b), (b, a)):
            if y['rank'] != 'correct':
                continue
            if x['rank'] == 'correct' and kind != 'cue_flip':
                stable[x['id']].setdefault(kind, y['id'])
            elif x['rank'] != 'correct' and (kind != 'cue_flip' or y['intent'] in x['competing_top_ids']):
                hard[x['id']].setdefault(kind, y['id'])

    # Group recordings of identical reference wording by speaker. Compare each
    # against a small, deterministic witness set, not every possible pair.
    for cohort in evidence['same_words']:
        roster = [uid for members in cohort for uid in members]
        if (len(cohort) < 2 or not all(cohort) or len(set(roster)) != len(roster)
                or not set(roster) <= eligible or len({by_id[x]['intent'] for x in roster}) != 1):
            raise ValueError('invalid same-wording audio cohort')
        picked = [(idx, next((uid for uid in members if observed[uid]['rank'] == 'correct'), None))
                  for idx, members in enumerate(cohort)]
        witnesses = [(idx, uid) for idx, uid in picked if uid][:2]
        for idx, members in enumerate(cohort):
            control = next((uid for j, uid in witnesses if j != idx), None)
            if control is not None:
                for uid in members:
                    compare('same_words_different_speaker', uid, control)

    for kind, left, right in evidence['local_pairs']:
        if kind not in {'cue_flip', 'cue_preserve'}:
            raise ValueError('unknown lexical contrast')
        compare(kind, left, right)

    schema = evidence.get('schema')
    full_train = schema == 'jev-paired-train-evidence-v3'
    if schema in {'jev-paired-train-evidence-v2','jev-paired-train-evidence-v3'}:
        channel_roster = set()
        for left, right in evidence['channel_pairs']:
            if (left in channel_roster or right in channel_roster
                    or by_id[left]['source_group'] != by_id[right]['source_group']):
                raise ValueError('channel pairs must be unique within the same utterance group')
            channel_roster.update((left,right))
            compare('same_utterance_different_microphone', left, right)

    def key(uid):
        return hashlib.sha256(f"{options['seed']}:{uid}".encode()).hexdigest(), uid

    replay_pool = defaultdict(list)
    for uid in stable:
        replay_pool[by_id[uid]['intent']].append(uid)
    if not hard or not replay_pool:
        raise ValueError('paired training requires both verified difficulty and replay support')
    for group in replay_pool.values():
        group.sort(key=key)
    fraction = Fraction(str(options['easy_fraction']))
    native_failures = {d['id'] for d in diagnostics if d['rank'] != 'correct'} if full_train else set(hard)
    probability_cases = {}
    if full_train:
        # Use disjoint top-loss tails among correctly ranked TRAIN audio, with
        # a fixed budget per aim; these do not certify a semantic root cause.
        for kind, metric in (('positive_underconfidence','positive_nll'), ('negative_overconfidence','negative_nll')):
            pool = (d for d in diagnostics if d['rank']=='correct' and d['id'] not in probability_cases)
            by_loss = sorted(pool, key=lambda d: (-d[metric], key(d['id'])))
            probability_cases.update((d['id'], kind) for d in by_loss[:options['probability_examples_per_type']])
    targeted = native_failures | probability_cases.keys()
    replay_pool = {k:[uid for uid in group if uid not in probability_cases] for k,group in replay_pool.items()}
    stable_replay = set(stable)-probability_cases.keys()
    target = min(len(stable_replay), max(options.get('min_unique_recordings', 0)-len(targeted),
        math.ceil(len(targeted) * fraction / (1-fraction))))
    # Spread replay across intents within the rounded sample budget.
    labels = sorted(replay_pool, key=key)
    prioritized = [replay_pool[label][depth] for depth in range(min(options['replay_class_priority'], target))
                   for label in labels if len(replay_pool[label]) > depth]
    replay = set(prioritized[:target])
    replay.update(sorted(stable_replay-replay, key=key)[:target-len(replay)])
    selected = sorted(targeted | replay, key=key)
    if len(selected) < options.get('min_unique_recordings',0):
        raise ValueError('verified TRAIN population cannot reach minimum unique recordings')
    by_class = defaultdict(Counter)
    by_bucket, by_kind, selected_types = Counter(), Counter(), Counter()
    for d in diagnostics:
        uid = d['id']
        bucket = ('paired_difficult' if uid in hard else 'paired_easy' if uid in stable else
                  'protected_or_not_in_isolated_pool' if uid not in eligible else
                  'unverified_rank_candidate' if d['rank'] != 'correct' else 'unclassified')
        d.update(bucket=bucket, selected=uid in targeted or uid in replay)
        if uid in hard or uid in stable:
            d['comparison_ids'] = hard[uid] if uid in hard else stable[uid]
        by_kind.update(hard.get(uid, {}).keys())
        by_bucket[bucket] += 1
        by_class[d['intent']].update(['population', bucket])
        if d['selected']:
            selected_kind = ('selected_difficult' if uid in hard else
                             'selected_rank_only' if uid in native_failures else
                             'selected_probability' if uid in probability_cases else 'selected_easy')
            by_class[d['intent']].update(['selected', selected_kind])
            if full_train:
                if uid in probability_cases:
                    kind = probability_cases[uid]
                elif uid in hard:
                    kind = next(k for k in ('cue_flip','cue_preserve','same_utterance_different_microphone',
                        'same_words_different_speaker') if k in hard[uid])
                elif d['rank']=='tied':
                    kind = 'rank_tie_only'
                elif uid in native_failures:
                    scenario = by_id[uid]['scenario']
                    kind = ('rank_same_scenario' if any(x.startswith(scenario+'_') for x in d['competing_top_ids'])
                            else 'rank_cross_scenario')
                else:
                    kind = 'easy_replay'
                d['sampling_type'] = kind
                selected_types[kind] += 1
    quality = {}
    probability_fields = ('positive_nll', 'negative_nll', 'positive_brier', 'negative_brier')
    for name in ('all', 'correct', 'wrong', 'tied'):
        group = [r for r in diagnostics if name == 'all' or r['rank'] == name]
        quality[name] = dict(count=len(group), **{
            k: math.fsum(r[k] for r in group)/len(group) if group else None for k in probability_fields})
    summary = dict(population=len(rows), eligible_for_pairs=len(eligible), selected=len(selected),
        selected_difficult=len(hard), selected_easy=len(replay), actual_easy_fraction=len(replay)/len(selected),
        requested_easy_fraction=options['easy_fraction'], buckets=dict(by_bucket),
        rank_counts=dict(Counter(d['rank'] for d in diagnostics)), paired_comparisons=dict(pair_counts),
        hard_evidence_by_kind=dict(by_kind), by_class={k: dict(v) for k,v in by_class.items()},
        probability_quality=quality, probability_used_for_admission=False,
        information_insufficient_gold=0, composition_or_capacity_gold=0, layout_predictions_present=False)
    if schema in {'jev-paired-train-evidence-v2','jev-paired-train-evidence-v3'}:
        summary.update(holdout_keys=evidence['holdout_keys'],
            channel_conflicts_excluded=evidence['channel_conflicts_excluded'],
            hard_source_groups=len({by_id[uid]['source_group'] for uid in hard}),
            selected_source_groups=len({by_id[uid]['source_group'] for uid in selected}),
            hard_evidence_overlap=dict(Counter('+'.join(sorted(kinds)) for kinds in hard.values())))
    if full_train:
        summary.update(selected_rank_only=len(native_failures-set(hard)),
            selected_native_failures=len(native_failures), min_unique_recordings=options['min_unique_recordings'],
            selected_classes=sum(v['selected']>0 for v in by_class.values()),
            replay_classes=sum(v['selected_easy']>0 for v in by_class.values()), historical_probes_used=False,
            probability_used_for_admission=bool(probability_cases), selected_probability=len(probability_cases),
            selected_by_type=dict(selected_types))
    return selected, diagnostics, summary
