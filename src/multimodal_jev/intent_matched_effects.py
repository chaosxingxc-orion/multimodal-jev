"""Paired D/Q/P learning effects; structured inference is not structured training."""
import math
from .intent_learning_trajectory import summarize as summarize_trajectory


METRICS = ('final_accuracy', 'final_macro_f1_observed', 'final_expected_reward', 'final_nll')


def compare(arms, schedule):
    if set(arms) != {'D', 'Q', 'P'} or not schedule:
        raise ValueError('all three training arms and fixed source schedule required')
    size = len(schedule)+1
    if any(len(points) != size for points in arms.values()):
        raise ValueError('complete matched curves including initial policy required')
    base = arms['D'][0]['score']['base_id']
    initial_adapter = arms['D'][0]['score']['adapter_id']
    populations = {}; admissions = {}; values = {}; rows_by_arm = {}
    for arm, points in arms.items():
        if [p['update'] for p in points] != list(range(size)):
            raise ValueError('complete contiguous update sequence required')
        if points[0]['score']['adapter_id'] != initial_adapter:
            raise ValueError('same initial adapter required')
        wire = 'D' if arm == 'D' else 'Q'
        values[arm] = []; rows_by_arm[arm] = []
        for point in points:
            score = point['score']; payload = score['reports'][wire]; metrics = payload['metrics']
            if score['role'] != 'learning' or score['base_id'] != base:
                raise ValueError('same-base learning reports required')
            records = metrics['records']; rows = {r['id']:r for r in records}
            if not rows or len(rows) != len(records): raise ValueError('unique complete population required')
            gold = {k:r['gold'][-1] for k,r in rows.items()}
            if arm not in populations:
                populations[arm] = gold; admissions[arm] = payload['admission']
            if gold != populations[arm] or payload['admission'] != admissions[arm]:
                raise ValueError('fixed gold and admission required')
            if any(type(r['correct']) is not bool or r['correct'] != (r['prediction'][-1] == r['gold'][-1]) for r in records):
                raise ValueError('Final correctness differs from semantic predictions')
            m = {k:metrics[k] for k in METRICS}
            if any(type(v) not in (int,float) or not math.isfinite(v) for v in m.values()):
                raise ValueError('finite metrics required')
            if not math.isclose(m['final_accuracy'], sum(r['correct'] for r in records)/len(records), abs_tol=1e-12):
                raise ValueError('aggregate accuracy differs from paired rows')
            values[arm].append(m); rows_by_arm[arm].append(rows)
    if not populations['D'] == populations['Q'] == populations['P']:
        raise ValueError('same final labels across all training arms required')
    # Q/P share input and initialization: differences at step0 are an invalid comparison.
    if (arms['Q'][0]['score']['reports']['Q'] != arms['P'][0]['score']['reports']['Q']):
        raise ValueError('Q/P initial structured reports differ')
    if admissions['Q'] != admissions['P']: raise ValueError('Q/P admission differs')
    for batch in schedule:
        if not batch or len(set(batch)) != len(batch) or not set(batch) <= populations['D'].keys():
            raise ValueError('known distinct scheduled sources required')
    curve = []; source_uses = 0
    for step in range(size):
        if step: source_uses += len(schedule[step-1])
        changes = {a:{k:values[a][step][k]-values[a][0][k] for k in METRICS} for a in arms}
        effects = {}
        for name,left,right in [('question_decomposition','Q','D'),('process_credit','P','Q'),('total_structure','P','D')]:
            a,b = rows_by_arm[left][step], rows_by_arm[right][step]
            effects[name] = dict(left_arm=left,right_arm=right,
                final_difference={k:values[left][step][k]-values[right][step][k] for k in METRICS},
                change_difference={k:changes[left][k]-changes[right][k] for k in METRICS},
                left_only_correct=sum(a[i]['correct'] and not b[i]['correct'] for i in a),
                right_only_correct=sum(b[i]['correct'] and not a[i]['correct'] for i in a))
        curve.append(dict(update=step,source_uses=source_uses,
            absolute={a:values[a][step] for a in arms},change_from_initial=changes,effects=effects))
    trajectories = {arm:summarize_trajectory(
        [dict(update=p['update'],source_uses=p['source_uses'],metrics=p['absolute'][arm]) for p in curve],
        registered_updates=len(schedule),
        directions={k:-1 if k=='final_nll' else 1 for k in METRICS}) for arm in arms}
    return dict(role='learning',complete=True,recordings=len(populations['D']),updates=len(schedule),curve=curve,
        trajectories=trajectories,
        metric_direction={k:'lower' if k=='final_nll' else 'higher' for k in METRICS},
        scope='Complete matched learning curves only. Training-arm identity and budget require external run authentication. '
              'No held-out gain, confidence interval, equal-compute or multihop-reasoning claim. No best-point selection.')
