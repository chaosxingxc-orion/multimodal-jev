"""Same-audio external FSC reference; different training conditions explicit."""
from collections import Counter,defaultdict
import random
from .fsc_published import parse_prediction,FIELDS
from .fsc_transfer_report import command,interval


def compare_native(task,roster,targets,predictions,policies,*,seed=941,replicates=2000):
    """Caller authenticates entire matrix and published run before reading gold.

    External output is one native semantic decision, never six binding views.
    Invalid/out-of-ontology outputs remain wrong in the full denominator.
    """
    if type(seed)!=int or type(replicates)!=int or replicates<100:raise ValueError('fixed bootstrap required')
    expected={'step0'}|{f'{a}-step{s}' for a in ('D','Q','P') for s in (6,12)}
    if set(policies)!=expected:raise ValueError('complete seven-policy matrix required')
    sources={r['id']:r for r in roster};gold={r['id']:r['gold'] for r in targets}
    reference={'fsc:'+r['example_id']:r for r in predictions}
    if (not sources or len(sources)!=len(roster) or len(gold)!=len(targets) or len(reference)!=len(predictions)
            or sources.keys()!=gold.keys() or sources.keys()!=reference.keys()):
        raise ValueError('same complete distinct recording population required')
    support={command(c['values']) for c in task['commands']}
    if len(support)!=31:raise ValueError('full FSC command ontology required')
    counts=Counter();predicted=Counter();correct=Counter();groups=defaultdict(list);truth={};invalid=0;components=Counter()
    for uid,source in sources.items():
        y=command(gold[uid]);r=reference[uid]
        if y not in support or source['audio_sha256']!=r['audio_sha256']:raise ValueError('reference target or audio differs')
        speaker=source['speaker_id']
        if type(speaker)!=str or not speaker:raise ValueError('speaker identity required')
        groups[speaker].append(uid);counts[y]+=1
        value=parse_prediction(r['raw_prediction']);p=command(value) if value is not None else None
        valid=p in support;invalid+=not valid;truth[uid]=p==y
        if valid:predicted[p]+=1
        if truth[uid]:correct[y]+=1
        if value is not None:
            for f in FIELDS:components[f]+=value[f]==gold[uid][f]
    speakers=sorted(groups)
    if len(speakers)<2:raise ValueError('at least two speaker clusters required')
    f1=lambda k:2*correct[k]/(counts[k]+predicted[k]) if counts[k]+predicted[k] else 0.
    n=len(sources)
    baseline=dict(recordings=n,correct=sum(truth.values()),accuracy=sum(truth.values())/n,
        invalid_or_out_of_ontology=invalid,observed_classes=len(counts),
        macro_f1_observed=sum(f1(k) for k in counts)/len(counts),macro_f1_full_ontology=sum(f1(k) for k in support)/len(support),
        component_accuracy={f:components[f]/n for f in FIELDS},
        per_class=[dict(command=dict(zip(FIELDS,k)),support=counts[k],predictions=predicted[k],correct=correct[k],f1=f1(k)) for k in sorted(support)],
        per_speaker={s:dict(recordings=len(ids),correct=sum(truth[i] for i in ids),accuracy=sum(truth[i] for i in ids)/len(ids)) for s,ids in sorted(groups.items() )})
    comparisons={};deltas={}
    for policy_key,policy in policies.items():
        for template in ('direct','structured'):
            cell=policy['cells'][template+'::native'];rows={r['id']:r for r in cell['records']}
            if len(rows)!=len(cell['records']) or rows.keys()!=sources.keys():raise ValueError('complete native Jev population required')
            for uid,r in rows.items():
                if r['gold']!=gold[uid] or r['speaker_id']!=sources[uid]['speaker_id']:raise ValueError('paired gold/speaker differs')
                p=command(r['semantic_prediction']['final'])
                if p not in support or type(r['final_correct'])!=bool or r['final_correct']!=(p==command(gold[uid])):
                    raise ValueError('Jev native semantic correctness differs')
            actual=sum(r['final_correct'] for r in rows.values())/n
            if cell['recordings']!=n or abs(cell['final_accuracy']-actual)>1e-12:raise ValueError('native aggregate differs')
            key=policy_key+'::'+template;deltas[key]={i:int(rows[i]['final_correct'])-int(truth[i]) for i in sources}
            comparisons[key]=dict(jev_accuracy=actual,reference_accuracy=baseline['accuracy'],
                accuracy_difference_jev_minus_reference=actual-baseline['accuracy'],
                jev_only_correct=sum(rows[i]['final_correct'] and not truth[i] for i in sources),
                reference_only_correct=sum(truth[i] and not rows[i]['final_correct'] for i in sources),
                both_correct=sum(rows[i]['final_correct'] and truth[i] for i in sources),
                both_wrong=sum(not rows[i]['final_correct'] and not truth[i] for i in sources))
    rng=random.Random(seed);samples={k:[] for k in deltas};baseline_samples=[]
    totals={k:{s:sum(v[i] for i in ids) for s,ids in groups.items()} for k,v in deltas.items()}
    correct_by_speaker={s:sum(truth[i] for i in ids) for s,ids in groups.items()}
    for _ in range(replicates):
        draw=[rng.choice(speakers) for _ in speakers];den=sum(len(groups[s]) for s in draw)
        baseline_samples.append(sum(correct_by_speaker[s] for s in draw)/den)
        for k in samples:samples[k].append(sum(totals[k][s] for s in draw)/den)
    baseline['accuracy_percentile95']=interval(baseline_samples)
    for k,v in samples.items():comparisons[k]['difference_percentile95']=interval(v)
    return dict(reference=baseline,native_comparisons=comparisons,bootstrap=dict(seed=seed,replicates=replicates,speaker_clusters=len(speakers)),
        scope='Same audio population, different training/model/output conditions. SpeechBrain FSC-task-trained reference versus SLURP-trained Jev transfer. Native semantics only; no external binding-view scores, matched-budget algorithm claim, checkpoint selection or training-seed uncertainty.')
