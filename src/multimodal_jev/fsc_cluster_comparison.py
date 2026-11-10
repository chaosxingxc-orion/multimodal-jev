"""Paired speaker-cluster intervals for complete six-view frozen evaluations."""
from collections import defaultdict

from .compare_policies import bootstrap
from .generative_teacher_feedback import prediction_status
from .six_view_score import score_six_views,VIEWS


def compare(requests,targets,parent,candidate,*,draws=10000,seed=17):
    if type(draws) is not int or draws<100 or type(seed) is not int:
        raise ValueError('at least 100 draws and integer seed required')
    if any(type(p.get('replicate')) is not int or p['replicate']!=0 for p in parent+candidate):
        raise ValueError('single deterministic replicate zero required')
    # The existing strict scorer verifies full views, definitions, input binding,
    # duplicates, malformed outputs and complete predictions for each prompt.
    base_score=score_six_views(requests,targets,parent)
    candidate_score=score_six_views(requests,targets,candidate)
    gold={t['id']:t for t in targets};left={p['example_id']:p for p in parent};right={p['example_id']:p for p in candidate}
    buckets=defaultdict(list);groups=defaultdict(dict);speaker_by_audio={}
    for row in requests:
        uid=row['id'];target=gold[uid];speaker=target['metadata'].get('speaker_id')
        if type(speaker) is not str or not speaker:raise ValueError('original speaker identity required; no fallback')
        audio=row['audio_sha256']
        if audio in speaker_by_audio and speaker_by_audio[audio]!=speaker:
            raise ValueError('same audio assigned conflicting speakers')
        speaker_by_audio[audio]=speaker
        a=prediction_status(row,left[uid],target);b=prediction_status(row,right[uid],target)
        view=target['metadata']['variant']
        for metric in ('correct','format_valid','schema_valid'):
            buckets[view+'/'+metric].append((speaker,bool(a[metric]),bool(b[metric])))
        groups[row['source_group']][view]=(speaker,bool(a['correct']),bool(b['correct']))
    for views in groups.values():
        if set(views)!=set(VIEWS) or len({v[0] for v in views.values()})!=1:
            raise ValueError('one speaker and all views per source recording required')
        speaker=views['native'][0]
        for a,b,name in (('native','permutation','permutation'),('native','rebind','rebind'),
                         ('native','missing_candidate','missing_candidate'),('noul.True','noul.False','noul')):
            buckets['pair/'+name].append((speaker,views[a][1] and views[b][1],views[a][2] and views[b][2]))
        buckets['all_six'].append((speaker,all(v[1] for v in views.values()),all(v[2] for v in views.values())))
    reports={}
    for metric,rows in sorted(buckets.items()):
        result=bootstrap(rows,draws,seed)
        result.update(both_correct=sum(a and b for _,a,b in rows),both_wrong=sum(not a and not b for _,a,b in rows),
            fixed=sum(not a and b for _,a,b in rows),regressed=sum(a and not b for _,a,b in rows))
        # The same seed and sorted speaker roster use the same cluster resample
        # across outcomes, preserving correlated views without pooling them.
        reports[metric]=result
    return dict(schema='fsc-six-view-speaker-cluster-comparison-v1',source_groups=len(groups),
        speakers=len(set(speaker_by_audio.values())),requests=len(requests),draws=draws,seed=seed,
        comparisons=reports,rescored_parent=base_score,rescored_candidate=candidate_score,
        estimand='Recording-weighted rate difference (candidate minus parent), paired percentile bootstrap '
        'resampling speakers with replacement and all their recordings together.',
        scope='Separate correlated view, correctness-pair and all-six outcomes; not pooled independent trials. '
        'Descriptive marginal 95% intervals, no multiple-testing correction or p-value. '
        'One frozen prompt pair/seed, not uncertainty over teacher selection or model training. '
        'Fewer than two speakers yields no interval; homogeneous empirical deltas can yield degenerate intervals.')
