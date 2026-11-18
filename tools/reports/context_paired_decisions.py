"""Descriptive paired decisions; source groups are not independent recordings."""
from collections import defaultdict
import math


def summarize(records):
    """Records contain aligned gold, source group, and two full probability lists."""
    if not records or len({r['id'] for r in records}) != len(records):
        raise ValueError('Nonempty unique paired recordings required')
    groups = defaultdict(list)
    size = len(records[0]['initial'])
    if not size:
        raise ValueError('Nonempty ontology required')
    for row in records:
        if (type(row['source_group']) is not str or not row['source_group']
                or type(row['gold']) is not int or not 0 <= row['gold'] < size):
            raise ValueError('Explicit group and gold index required')
        values = []
        for name in ('initial', 'selected'):
            p = row[name]
            if (len(p) != size or any(type(v) not in (int, float) or not math.isfinite(v)
                                     or not 0 <= v <= 1 for v in p)):
                raise ValueError('Aligned finite full probability vectors required')
            chosen = [i for i, v in enumerate(p) if v >= .5]
            hit = row['gold'] in chosen
            values.append(dict(top1=int(max(range(size), key=lambda i: p[i]) == row['gold']),
                               exact_set=int(chosen == [row['gold']]),
                               set_f1=2 * int(hit) / (len(chosen) + 1),
                               selected_count=len(chosen), empty=int(not chosen),
                               multiple=int(len(chosen) > 1)))
        groups[row['source_group']].append((row['gold'], values))
    if any(len({gold for gold, _ in rows}) != 1 for rows in groups.values()):
        raise ValueError('Conflicting gold labels in source group')
    output = {}
    for metric in ('top1', 'exact_set', 'set_f1', 'selected_count', 'empty', 'multiple'):
        grouped = [[(v[0][metric], v[1][metric]) for _, v in rows] for rows in groups.values()]
        pairs = [p for group in grouped for p in group]
        left = sum(a for a, _ in pairs) / len(pairs)
        right = sum(b for _, b in pairs) / len(pairs)
        group_left = sum(sum(a for a, _ in g) / len(g) for g in grouped) / len(grouped)
        group_right = sum(sum(b for _, b in g) / len(g) for g in grouped) / len(grouped)
        result = dict(recording_weighted=dict(initial=left, selected=right, delta=right-left),
                      source_group_macro=dict(initial=group_left, selected=group_right,
                                              delta=group_right-group_left),
                      recordings_increased=sum(b > a for a, b in pairs),
                      recordings_decreased=sum(b < a for a, b in pairs),
                      groups_with_mixed_directions=sum(any(b > a for a, b in g)
                                                       and any(b < a for a, b in g) for g in grouped))
        if metric in ('top1', 'exact_set'):
            result['transition_counts'] = dict(both_correct=sum(a == b == 1 for a, b in pairs),
                                               both_wrong=sum(a == b == 0 for a, b in pairs),
                                               repaired=sum(a == 0 and b == 1 for a, b in pairs),
                                               regressed=sum(a == 1 and b == 0 for a, b in pairs))
        output[metric] = result
    sizes = [len(rows) for rows in groups.values()]
    return dict(recordings=len(records), source_groups=len(groups),
                recordings_per_group=dict(min=min(sizes), max=max(sizes)), metrics=output,
                scope='Descriptive paired diagnostic at fixed threshold 0.5. Source-group macro is a different '
                      'estimand, not a replacement for the predeclared recording-weighted primary metric. '
                      'No independence, confidence interval, training-seed or pure-RL causal claim.')
