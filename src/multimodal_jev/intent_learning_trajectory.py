"""Describe every registered curve point; never select the best checkpoint."""
import math


def summarize(points, *, registered_updates, directions):
    """Trapezoidal areas use cumulative source uses, not independent sample count.

    A contiguous prefix is allowed but explicitly incomplete. Its normalized
    area covers only observed exposure and must not be compared as a full budget.
    Directions are +1 for higher-is-better and -1 for lower-is-better.
    """
    if (type(registered_updates) is not int or registered_updates < 1 or not points
            or not directions or any(type(d) is not int or d not in (-1, 1) for d in directions.values())
            or len(points) > registered_updates+1):
        raise ValueError('registered positive budget, metric directions and nonempty prefix required')
    previous = None
    for step, point in enumerate(points):
        if (type(point['update']) is not int or point['update'] != step
                or type(point['source_uses']) is not int
                or (step == 0 and point['source_uses'] != 0)
                or (step > 0 and point['source_uses'] <= previous['source_uses'])
                or set(point['metrics']) != set(directions)
                or any(type(v) not in (int, float) or not math.isfinite(v)
                       for v in point['metrics'].values())):
            raise ValueError('contiguous finite curve with increasing exposure required')
        previous = point
    exposure = points[-1]['source_uses']
    metrics = {}
    for name, direction in directions.items():
        values = [p['metrics'][name] for p in points]
        gains = [direction*(v-values[0]) for v in values]
        changes = [direction*(b-a) for a,b in zip(values, values[1:])]
        area = sum((b['source_uses']-a['source_uses'])*(x+y)/2
                   for a,b,x,y in zip(points, points[1:], values, values[1:]))
        gain_area = sum((b['source_uses']-a['source_uses'])*(x+y)/2
                        for a,b,x,y in zip(points, points[1:], gains, gains[1:]))
        metrics[name] = dict(initial=values[0], final=values[-1],
            final_change=values[-1]-values[0], direction=direction,
            final_directional_gain=gains[-1], exposure_area=area,
            mean_over_observed_exposure=area/exposure if exposure else None,
            mean_directional_gain=gain_area/exposure if exposure else None,
            improving_transitions=sum(x > 0 for x in changes),
            worsening_transitions=sum(x < 0 for x in changes),
            unchanged_transitions=sum(x == 0 for x in changes),
            checkpoints_below_initial=sum(x < 0 for x in gains[1:]),
            worst_directional_gain=min(gains))
    return dict(registered_updates=registered_updates, observed_updates=len(points)-1,
        complete=len(points)==registered_updates+1, source_uses=exposure, metrics=metrics,
        scope='Descriptive full-prefix learning trajectory, trapezoidal interpolation over source uses. '
              'No best-checkpoint selection, significance, independent-evaluation or equal-compute claim. '
              'An incomplete prefix is not a completed registered-budget result.')
