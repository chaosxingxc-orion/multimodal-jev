"""Strict paired Jev comparisons, retaining semantic-group dependence."""
from collections import Counter

from .answers import parse_answer,AnswerError
from .slurp_jev_score import score_bundle
from .task_schema import DecisionSchema


def paired_bundle(bundle,target,bank,parent,candidate):
    # Full identity, request hash, coverage and typed gold validation precedes
    # any comparison. Malformed JSON is never a valid empty list or null.
    a_counts,_=score_bundle(bundle,target,bank,parent)
    b_counts,_=score_bundle(bundle,target,bank,candidate)
    aa={r['example_id']:r for r in parent};bb={r['example_id']:r for r in candidate}
    counts={};all_a=[];all_b=[]
    for key,definition in bank['tasks'].items():
        uid=bundle['id']+'::'+key;schema=DecisionSchema.from_dict(definition);gold=target['values'][key]
        def correct(raw):
            try:value=parse_answer(raw)['value']
            except AnswerError:return False
            return schema.accepts(value) and type(value) is type(gold) and value==gold
        a=correct(aa[uid]['raw_text']);b=correct(bb[uid]['raw_text'])
        all_a.append(a);all_b.append(b)
        labels=[key]
        if key.startswith('entity.'):
            # Aggregate by question family, retaining present/absent support.
            family=key.rsplit('.',1)[-1];prefix=key.rsplit('.',1)[0]
            support='nonempty' if target['values'][prefix+'.values'] else 'empty'
            labels += ['family.'+family, 'family.'+family+'.'+support]
        for label in labels:
            counts.setdefault(label,Counter())[transition(a,b)]+=1
    return counts,(all(all_a),all(all_b)),a_counts,b_counts


def transition(a,b):
    return 'both_correct' if a and b else 'both_wrong' if not a and not b else 'fixed' if b else 'regressed'


class PairedAccumulator:
    def __init__(self):
        self.by_task={};self.by_group={};self.recordings=Counter();self.seen=set()

    def add(self,bundle,target,bank,parent,candidate):
        if bundle['id'] in self.seen:raise ValueError('duplicate paired recording')
        counts,whole,_,_=paired_bundle(bundle,target,bank,parent,candidate)
        self.seen.add(bundle['id'])
        for key,cell in counts.items():self.by_task.setdefault(key,Counter()).update(cell)
        self.recordings[transition(*whole)]+=1
        group=self.by_group.setdefault(bundle['source_group'],[True,True])
        group[0]=group[0] and whole[0];group[1]=group[1] and whole[1]

    def report(self):
        groups=Counter(transition(*pair) for pair in self.by_group.values())
        def cell(counts):
            return {key:counts.get(key,0) for key in ('both_correct','both_wrong','fixed','regressed')}
        return dict(by_task={key:cell(value) for key,value in sorted(self.by_task.items())},
            all_tasks_by_recording=cell(self.recordings),all_tasks_all_recordings_by_source_group=cell(groups),
            recordings=len(self.seen),source_groups=len(self.by_group),
            scope='Paired descriptive counts; correlated tasks and recordings are not independent significance samples')


def require_full_roster(plan,prompts,runs):
    """Reject incomplete comparison before reading any task targets."""
    shard_ids=[s['id'] for s in plan['shards']]
    if len(shard_ids)!=len(set(shard_ids)) or not shard_ids:
        raise ValueError('unique nonempty full shard plan required')
    ids=[uid for s in plan['shards'] for uid in s['ordered_bundle_ids']]
    groups=[g for s in plan['shards'] for g in s['source_groups']]
    if (len(ids)!=len(set(ids)) or len(ids)!=plan['recordings']
            or len(groups)!=len(set(groups)) or len(groups)!=plan['source_groups']
            or sum(s['decisions'] for s in plan['shards'])!=plan['decisions']
            or any(s['recordings']!=len(s['ordered_bundle_ids'])
                   or s['decisions']!=s['recordings']*plan['tasks_per_recording'] for s in plan['shards'])):
        raise ValueError('complete disjoint recording and group coverage required')
    if set(runs)!=set(prompts) or any(set(runs[key])!=set(shard_ids) for key in prompts):
        raise ValueError('every registered prompt and shard must complete')
