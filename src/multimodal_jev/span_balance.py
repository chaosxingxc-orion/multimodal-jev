"""Learning-only exact joint-label balancing for finite Span debug batches."""
from collections import defaultdict
import random


def balanced_binary_batches(bundles, seed):
    """Uniform observed full Boolean signatures; pair complementary labels.

    No synthetic labels, repeats or validation access. Incomplete supervision is
    excluded only from this debug arm, never changed in the source release.
    """
    if not bundles:raise ValueError('nonempty bundles required')
    width=len(bundles[0]['targets'])
    if width<1:raise ValueError('nonempty target tuple required')
    buckets=defaultdict(list)
    for i,b in enumerate(bundles):
        if len(b['targets'])!=width or len(b['supervised'])!=width:
            raise ValueError('uniform Span shape required')
        if not all(b['supervised']):continue
        if any(type(v) is not bool for v in b['targets']):raise ValueError('binary task required')
        buckets[tuple(b['targets'])].append(i)
    if len(buckets)!=2**width:
        raise ValueError('all joint Boolean signatures required; never fabricate missing classes')
    minimum=min(map(len,buckets.values()))
    rng=random.Random(seed)
    selected={}
    for signature,indices in sorted(buckets.items()):
        indices=sorted(indices,key=lambda i:bundles[i]['id'])
        rng.shuffle(indices)
        selected[signature]=indices[:minimum]
    batches=[]
    for signature in sorted(selected):
        complement=tuple(not v for v in signature)
        if signature>complement:continue
        batches.extend([[a,b] for a,b in zip(selected[signature],selected[complement],strict=True)])
    rng.shuffle(batches)
    for batch in batches:rng.shuffle(batch)
    used={i for batch in batches for i in batch}
    report=dict(mode='joint_uniform_complement_pairs_v1',seed=seed,
        input_recordings=len(bundles),selected_recordings=len(used),updates=len(batches),
        per_signature=minimum,signatures=len(selected),
        per_span_counts=[{'false':len(used)//2,'true':len(used)//2} for _ in range(width)],
        excluded=[dict(id=b['id'],reason='incomplete_supervision' if not all(b['supervised']) else 'joint_stratum_downsample')
                  for i,b in enumerate(bundles) if i not in used],
        batch_recording_ids=[[bundles[i]['id'] for i in batch] for batch in batches],
        scope='Learning-only balanced debug arm; official and evaluation distributions unchanged.')
    return batches,report
