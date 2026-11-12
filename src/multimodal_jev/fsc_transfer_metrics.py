"""Paired FSC decision metrics across frozen templates and policy checkpoints.

Caller authenticates backend artifacts, split admission and checkpoint hashes.
This scorer validates exact input/label populations; it never repairs Final.
"""
import json
import math
from .slurp_training import content_hash
from .fsc_span_transfer import compile_request,score_one,FIELDS

TEMPLATES=('direct','structured')
VARIANTS=('native','permutation','rebind')


def score_policy(task,roster,targets,records,*,policy_id,seed=317):
    if not roster or len({r['id'] for r in roster})!=len(roster):raise ValueError('distinct nonempty roster required')
    ids={r['id'] for r in roster};gold={r['id']:r['gold'] for r in targets}
    if len(gold)!=len(targets) or set(gold)!=ids:raise ValueError('complete distinct target population required')
    expected={r['id']+'::'+t+'::'+v for r in roster for t in TEMPLATES for v in VARIANTS}
    by_id={r['id']:r for r in records}
    if len(by_id)!=len(records) or set(by_id)!=expected:raise ValueError('complete six-view prediction population required')
    cells={};scored={}
    for template in TEMPLATES:
        for variant in VARIANTS:
            rows=[]
            for source in roster:
                uid=source['id']+'::'+template+'::'+variant;record=by_id[uid]
                request,binding=compile_request(source['audio'],task,template,variant=variant,seed=seed)
                identity=record['request_identity']
                if (identity['request']!=json.loads(json.dumps(request.model_input()))
                        or record['policy_id']!=policy_id or record['audio_sha256']!=source['audio_sha256']):
                    raise ValueError('exact configured input, audio and policy required')
                logps=record['candidate_logprobs']
                if len(logps)!=len(request.questions):raise ValueError('complete Span support required')
                prediction={};final_logp=None
                for question,logp in zip(request.questions,logps,strict=True):
                    if (len(logp)!=len(question.values) or any(type(v) not in (int,float) or not math.isfinite(v) for v in logp)
                            or abs(sum(math.exp(v) for v in logp)-1)>1e-5):
                        raise ValueError('complete normalized finite candidate probabilities required')
                    prediction[question.id]=question.values[max(range(len(logp)),key=lambda k:logp[k])]
                    if question.id=='final':final_logp=logp
                result=score_one(prediction,binding,gold[source['id']])
                target_id=next(k for k,v in binding['final'].items() if v==gold[source['id']])
                y=request.questions[-1].values.index(target_id)
                rows.append(dict(id=source['id'],speaker_id=source['speaker_id'],gold=gold[source['id']],
                    final_nll=-final_logp[y],gold_probability=math.exp(final_logp[y]),**result))
            n=len(rows);key=template+'::'+variant;scored[key]={r['id']:r for r in rows}
            cells[key]=dict(recordings=n,final_accuracy=sum(r['final_correct'] for r in rows)/n,
                final_nll=sum(r['final_nll'] for r in rows)/n,final_expected_reward=sum(r['gold_probability'] for r in rows)/n,
                final_component_accuracy={f:sum(r['final_component_correct'][f] for r in rows)/n for f in FIELDS},
                process_component_accuracy={f:sum(r['process_correct'][f] for r in rows)/n for f in FIELDS} if template=='structured' else None,
                compatibility=sum(r['compatible'] for r in rows)/n if template=='structured' else None,records=rows)
    robustness={}
    for template in TEMPLATES:
        native=scored[template+'::native']
        for variant in VARIANTS[1:]:
            changed=scored[template+'::'+variant]
            robustness[template+'::'+variant]=dict(
                both_correct=sum(native[i]['final_correct'] and changed[i]['final_correct'] for i in ids)/len(ids),
                semantic_agreement=sum(native[i]['semantic_prediction']['final']==changed[i]['semantic_prediction']['final'] for i in ids)/len(ids))
    return dict(policy_id=policy_id,seed=seed,task_sha256=content_hash(task),
        roster_sha256=content_hash(sorted(roster,key=lambda r:r['id'])),cells=cells,robustness=robustness,
        scope='Provided complete FSC population only. Compatibility/agreement can be high when wrong; no dataset admission or efficacy claim from scorer execution.')


def compare_checkpoints(before,after):
    if any(before[k]!=after[k] for k in ('task_sha256','roster_sha256')):
        raise ValueError('same frozen task and audio roster required')
    if before['seed']!=after['seed'] or set(before['cells'])!=set(after['cells']):raise ValueError('same frozen views required')
    expected={t+'::'+v for t in TEMPLATES for v in VARIANTS}
    if set(before['cells'])!=expected:raise ValueError('complete fixed six-view matrix required')
    changes={};population=None
    for key in sorted(expected):
        a=before['cells'][key];b=after['cells'][key]
        left={r['id']:r for r in a['records']};right={r['id']:r for r in b['records']}
        if (len(left)!=len(a['records']) or len(right)!=len(b['records']) or left.keys()!=right.keys()
                or any(left[i]['gold']!=right[i]['gold'] or left[i]['speaker_id']!=right[i]['speaker_id'] for i in left)):
            raise ValueError('same complete paired source, speaker and target population required')
        identity={i:(r['speaker_id'],r['gold']) for i,r in left.items()}
        if population is not None and identity!=population:raise ValueError('view populations differ')
        population=identity
        for cell in (a,b):
            rows=cell['records']
            if (not rows or cell['recordings']!=len(rows)
                    or not math.isclose(cell['final_accuracy'],sum(r['final_correct'] for r in rows)/len(rows),abs_tol=1e-12)
                    or not math.isclose(cell['final_nll'],sum(r['final_nll'] for r in rows)/len(rows),abs_tol=1e-12)):
                raise ValueError('aggregate metrics differ from complete records')
        changes[key]=dict(recordings=len(left),accuracy_delta=b['final_accuracy']-a['final_accuracy'],
            wrong_to_right=sum(not left[i]['final_correct'] and right[i]['final_correct'] for i in left),
            right_to_wrong=sum(left[i]['final_correct'] and not right[i]['final_correct'] for i in left),
            nll_delta=b['final_nll']-a['final_nll'])
    interaction={}
    for variant in VARIANTS:
        d='direct::'+variant;s='structured::'+variant
        interaction[variant]=dict(before_template_gain=before['cells'][s]['final_accuracy']-before['cells'][d]['final_accuracy'],
            after_template_gain=after['cells'][s]['final_accuracy']-after['cells'][d]['final_accuracy'],
            rl_template_interaction=changes[s]['accuracy_delta']-changes[d]['accuracy_delta'])
    return dict(before_policy=before['policy_id'],after_policy=after['policy_id'],changes=changes,interaction=interaction,
        scope='Paired point estimates only; speaker-cluster uncertainty and source checkpoint selection must be supplied by the evaluation protocol. No causal reasoning attribution.')


def speaker_bootstrap(before,after,*,seed=941,replicates=2000):
    """Resample speakers with all their recordings, sharing draws across views."""
    import random
    from collections import defaultdict
    compare_checkpoints(before,after)
    if type(seed)!=int or type(replicates)!=int or replicates<100:
        raise ValueError('fixed seed and at least100 replicates required')
    groups=defaultdict(list)
    for row in before['cells']['direct::native']['records']:
        if not isinstance(row['speaker_id'],str) or not row['speaker_id']:
            raise ValueError('nonempty speaker identity required')
        groups[row['speaker_id']].append(row['id'])
    speakers=sorted(groups)
    if len(speakers)<2:raise ValueError('at least two speaker clusters required')
    keys=sorted(before['cells']);deltas={}
    for key in keys:
        a={r['id']:r for r in before['cells'][key]['records']};b={r['id']:r for r in after['cells'][key]['records']}
        if set(a)!={i for ids in groups.values() for i in ids}:raise ValueError('view populations differ')
        deltas[key]={i:int(b[i]['final_correct'])-int(a[i]['final_correct']) for i in a}
    samples={k:[] for k in keys};interactions={v:[] for v in VARIANTS};rng=random.Random(seed)
    for _ in range(replicates):
        ids=[i for _ in speakers for i in groups[rng.choice(speakers)]]
        means={k:sum(deltas[k][i] for i in ids)/len(ids) for k in keys}
        for k in keys:samples[k].append(means[k])
        for v in VARIANTS:interactions[v].append(means['structured::'+v]-means['direct::'+v])
    def interval(values):
        values=sorted(values)
        def q(p):
            at=(len(values)-1)*p;lo=int(at);hi=min(lo+1,len(values)-1)
            return values[lo]+(values[hi]-values[lo])*(at-lo)
        return [q(.025),q(.975)]
    return dict(seed=seed,replicates=replicates,speaker_clusters=len(speakers),
        accuracy_delta_intervals={k:interval(v) for k,v in samples.items()},
        interaction_intervals={k:interval(v) for k,v in interactions.items()},
        scope='Paired speaker-cluster percentile95 intervals for utterance-weighted accuracy; shared draws across all views. Does not cover cross-seed variation, repeated-command dependence across speakers or multiple comparisons.')
