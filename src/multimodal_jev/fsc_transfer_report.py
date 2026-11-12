"""Supplement complete frozen transfer scores; never select a checkpoint."""
from collections import Counter,defaultdict
import math
import random
from .fsc_span_transfer import FIELDS
from .fsc_transfer_metrics import compare_checkpoints,VARIANTS


def command(value):
    if not isinstance(value,dict) or set(value)!=set(FIELDS):raise ValueError('complete native command required')
    if any(type(value[f]) is not str or not value[f] for f in FIELDS):raise ValueError('named component values required')
    return tuple(value[f] for f in FIELDS)


def cell_report(cell,support):
    counts=Counter();predicted=Counter();correct=Counter();speakers=defaultdict(lambda:[0,0]);signatures=Counter()
    ids=set()
    for row in cell['records']:
        if row['id'] in ids:raise ValueError('duplicate recording')
        ids.add(row['id']);gold=command(row['gold']);prediction=command(row['semantic_prediction']['final'])
        if gold not in support or prediction not in support:raise ValueError('command outside frozen ontology')
        truth=gold==prediction
        if type(row['final_correct']) is not bool or row['final_correct']!=truth:raise ValueError('final correctness disagrees with semantic prediction')
        speaker=row['speaker_id']
        if type(speaker) is not str or not speaker:raise ValueError('speaker identity required')
        counts[gold]+=1;predicted[prediction]+=1;correct[gold]+=truth
        speakers[speaker][0]+=truth;speakers[speaker][1]+=1
        process=row['process_correct']
        if process is not None:
            if set(process)!=set(FIELDS) or any(type(v) is not bool for v in process.values()):raise ValueError('complete Boolean process scores required')
            if any(process[f]!=(row['semantic_prediction'][f]==row['gold'][f]) for f in FIELDS):raise ValueError('process correctness differs')
            signatures[str(int(all(process.values())))+str(int(truth))]+=1
    if not ids or cell['recordings']!=len(ids) or not math.isclose(cell['final_accuracy'],sum(correct.values())/len(ids),abs_tol=1e-12):
        raise ValueError('complete matching aggregate required')
    observed=sorted(counts);f1=lambda k:2*correct[k]/(counts[k]+predicted[k]) if counts[k]+predicted[k] else 0.
    return dict(recordings=len(ids),observed_classes=len(counts),ontology_classes=len(support),
        macro_f1_observed=sum(f1(k) for k in observed)/len(observed),
        macro_f1_full_ontology=sum(f1(k) for k in support)/len(support),
        macro_recall_observed=sum(correct[k]/counts[k] for k in observed)/len(observed),
        per_class=[dict(command=dict(zip(FIELDS,k)),support=counts[k],predictions=predicted[k],correct=correct[k],
            recall=correct[k]/counts[k] if counts[k] else None,f1=f1(k)) for k in sorted(support)],
        per_speaker={s:dict(correct=c,recordings=n,accuracy=c/n) for s,(c,n) in sorted(speakers.items())},
        process_final_counts={k:signatures[k] for k in ('00','01','10','11')} if signatures else None,
        process_all_correct_final_accuracy=(signatures['11']/(signatures['10']+signatures['11']) if signatures['10']+signatures['11'] else None))


def interval(values):
    values=sorted(values)
    def quantile(p):
        a=(len(values)-1)*p;i=int(a);j=min(i+1,len(values)-1)
        return values[i]+(values[j]-values[i])*(a-i)
    return [quantile(.025),quantile(.975)]


def summarize(task,policies,*,seed=941,replicates=2000):
    if type(seed) is not int or type(replicates) is not int or replicates<100:raise ValueError('fixed seed and at least100 draws required')
    if 'step0' not in policies or len(policies)<3:raise ValueError('complete matrix including initial and subsequent policies required')
    from .slurp_training import content_hash
    if any(p['task_sha256']!=content_hash(task) for p in policies.values()):raise ValueError('frozen task identity differs')
    support={command(c['values']) for c in task['commands']}
    if len(support)!=31:raise ValueError('full FSC ontology required')
    for policy in policies.values():compare_checkpoints(policies['step0'],policy)
    reports={k:{v:cell_report(c,support) for v,c in p['cells'].items()} for k,p in policies.items()}
    groups=defaultdict(list)
    for row in policies['step0']['cells']['direct::native']['records']:groups[row['speaker_id']].append(row['id'])
    speakers=sorted(groups)
    if len(speakers)<2:raise ValueError('at least two speakers required')
    values={}
    for key,policy in policies.items():
        for variant in VARIANTS:
            a={r['id']:r for r in policy['cells']['direct::'+variant]['records']}
            b={r['id']:r for r in policy['cells']['structured::'+variant]['records']}
            values[key,variant]={i:int(b[i]['final_correct'])-int(a[i]['final_correct']) for i in a}
    samples={k:[] for k in values};rng=random.Random(seed)
    # Shared speaker draws preserve all recording/view/policy pairing.
    totals={k:{s:sum(v[i] for i in ids) for s,ids in groups.items()} for k,v in values.items()}
    for _ in range(replicates):
        draw=[rng.choice(speakers) for _ in speakers];n=sum(len(groups[s]) for s in draw)
        for key in values:samples[key].append(sum(totals[key][s] for s in draw)/n)
    gains={key:{variant:dict(accuracy_gain=sum(values[key,variant].values())/sum(map(len,groups.values())),
        percentile95=interval(samples[key,variant])) for variant in VARIANTS} for key in policies}
    return dict(policies=reports,template_gains=gains,bootstrap=dict(seed=seed,replicates=replicates,speaker_clusters=len(speakers)),
        scope='Supplementary full-matrix diagnostics. Shared speaker-cluster draws, utterance-weighted template effects; no training-seed, cross-speaker repeated-command or multiplicity coverage. No selection, answer repair or causal reasoning claim.')
