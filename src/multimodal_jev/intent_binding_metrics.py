"""Learning-only task metrics across fixed candidate-binding views.

Each recording contributes once to source-level robustness. Four derived views
are not independent recordings. Caller authenticates the full backend gate.
"""
import math
from .intent_policy_metrics import score as score_intent
from .span_candidate_intervention import VARIANTS
from .span_decision_margin import measure,summarize


def score(rows,records,arm):
    if arm not in ('D','Q') or not rows:raise ValueError('nonempty D/Q wire population required')
    groups={};indexed={r['id']:r for r in records}
    if len(indexed)!=len(records) or len({r['gate_id'] for r in rows})!=len(rows):
        raise ValueError('unique full view population required')
    if set(indexed)!={r['gate_id'] for r in rows}:raise ValueError('exact full view population required')
    for row in rows:
        group=groups.setdefault(row['source_id'],{})
        if row['variant'] in group:raise ValueError('duplicate source variant')
        group[row['variant']]=row
    if any(set(g)!=set(VARIANTS) for g in groups.values()):raise ValueError('all four variants required for each source')
    reports={};predictions={};ranks={}
    for variant in VARIANTS:
        subset=[g[variant] for g in groups.values()]
        # Adapter only renames record IDs; probabilities remain in supplied order.
        items=[dict(row,id=row['source_id']) for row in subset]
        wire=[dict(indexed[row['gate_id']],id=row['source_id']+'::'+arm) for row in subset]
        metrics=score_intent(items,wire,arm)
        ranking=[dict(id=row['source_id'],**measure(indexed[row['gate_id']]['candidate_logprobs'][-1],
            row['target']['targets'][-1])) for row in subset]
        reports[variant]=dict(metrics=metrics,ranking=summarize(ranking))
        predictions[variant]={r['id']:r for r in metrics['records']}
        ranks[variant]={r['id']:r for r in ranking}
    source_records=[]
    for uid,group in groups.items():
        native=group['native'];gold=native['target']['target_values']
        canonical_values=[set(q.values) for q in native['request'].questions]
        for variant,row in group.items():
            if (row['target']['target_values']!=gold or row['request'].audio!=native['request'].audio
                    or row['audio_sha256']!=native['audio_sha256']
                    or row['target']['request_weight']!=native['target']['request_weight']
                    or [set(q.values) for q in row['request'].questions]!=canonical_values):
                raise ValueError('views must preserve source, semantic gold, support and weight')
        by_view={v:predictions[v][uid] for v in VARIANTS}
        semantic={v:by_view[v]['prediction'][-1] for v in VARIANTS}
        final_gold=gold[-1]
        if any(by_view[v]['gold']!=gold for v in VARIANTS):raise ValueError('reviewed semantic gold differs')
        all_correct=all(by_view[v]['correct'] for v in VARIANTS)
        source_records.append(dict(id=uid,gold=final_gold,predictions=semantic,
            correct={v:by_view[v]['correct'] for v in VARIANTS},all_views_correct=all_correct,
            any_view_correct=any(by_view[v]['correct'] for v in VARIANTS),
            all_views_same_semantic=len(set(semantic.values()))==1,
            minimum_gold_probability=min(by_view[v]['gold_probability'][-1] for v in VARIANTS),
            mean_gold_probability=sum(by_view[v]['gold_probability'][-1] for v in VARIANTS)/len(VARIANTS),
            ranks={v:ranks[v][uid]['gold_rank'] for v in VARIANTS}))
    pairs={}
    for v in VARIANTS[1:]:
        pairs[v]=dict(recordings=len(groups),
            both_correct=sum(r['correct']['native'] and r['correct'][v] for r in source_records),
            native_only_correct=sum(r['correct']['native'] and not r['correct'][v] for r in source_records),
            transformed_only_correct=sum(not r['correct']['native'] and r['correct'][v] for r in source_records),
            same_semantic=sum(r['predictions']['native']==r['predictions'][v] for r in source_records))
    n=len(groups)
    return dict(recordings=n,view_requests=len(rows),variants=reports,native_pairs=pairs,
        all_views_correct=sum(r['all_views_correct'] for r in source_records),
        all_views_correct_rate=sum(r['all_views_correct'] for r in source_records)/n,
        all_views_same_semantic_rate=sum(r['all_views_same_semantic'] for r in source_records)/n,
        all_views_same_but_wrong=sum(r['all_views_same_semantic'] and not r['all_views_correct'] for r in source_records),
        any_view_correct_diagnostic=sum(r['any_view_correct'] for r in source_records),
        mean_minimum_gold_probability=sum(r['minimum_gold_probability'] for r in source_records)/n,
        records=source_records,scope='Fixed learning sources, all four views reported. Any-view correct is an oracle diagnostic, never a deployable score. No held-out/transfer or equal-compute claim.')


def compare(before,after,updated_sources):
    if before['recordings']!=after['recordings'] or before['view_requests']!=after['view_requests']:
        raise ValueError('matched source/view population required')
    left={r['id']:r for r in before['records']};right={r['id']:r for r in after['records']}
    if (len(left)!=len(before['records']) or len(right)!=len(after['records']) or left.keys()!=right.keys()
            or any(left[k]['gold']!=right[k]['gold'] for k in left)
            or len(set(updated_sources))!=len(updated_sources) or not set(updated_sources)<=left.keys()):
        raise ValueError('same semantic gold and distinct updated learning sources required')
    groups={}
    for name,ids in [('all_learning',list(left)),('updated',updated_sources),
        ('other_learning',[k for k in left if k not in updated_sources])]:
        groups[name]=dict(recordings=len(ids),variants={})
        for v in VARIANTS:
            a=sum(left[k]['correct'][v] for k in ids);b=sum(right[k]['correct'][v] for k in ids)
            groups[name]['variants'][v]=dict(before_correct=a,after_correct=b,
                accuracy_delta=(b-a)/len(ids) if ids else None,
                gained=sum(not left[k]['correct'][v] and right[k]['correct'][v] for k in ids),
                lost=sum(left[k]['correct'][v] and not right[k]['correct'][v] for k in ids))
        groups[name]['all_views_correct_delta']=(sum(right[k]['all_views_correct']-left[k]['all_views_correct'] for k in ids)/len(ids)) if ids else None
    return dict(groups=groups,scope='Paired same-learning-source changes; other_learning is not development. Fixed views, no template or checkpoint selection.')
