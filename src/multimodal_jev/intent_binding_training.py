"""Source-only binding augmentation with external, semantically remapped credit.

All views share one recording/source. Their weights partition its original
budget. This is a new training factor, never an in-place D/Q/P pilot change.
"""
import copy
import math
from .span_candidate_intervention import intervene, VARIANTS
from .slurp_training import content_hash


def expand(item, variants=VARIANTS):
    if tuple(variants) != VARIANTS:
        raise ValueError('fixed complete four-view augmentation required')
    original=item['request']; target=item['target']; n=len(original.questions)
    if any(len(target[k])!=n for k in ('targets','target_values','supervised','span_weights')):
        raise ValueError('complete external supervision layout required')
    weight=target['request_weight']
    if type(weight) not in (int,float) or not math.isfinite(weight) or weight<=0:
        raise ValueError('positive source weight required')
    for q,index,value,known in zip(original.questions,target['targets'],target['target_values'],target['supervised'],strict=True):
        if known is not True or type(index) is not int or not 0<=index<len(q.values) or q.values[index]!=value:
            raise ValueError('reviewed canonical target must match source values')
    expanded=[]
    for variant in variants:
        # Input transformation receives no supervision, source score or gold.
        request,binding=intervene(original,variant)
        uid=item['id']+'::binding-'+variant
        row=dict(id=uid,source_id=item['id'],variant=variant,audio_sha256=item['audio_sha256'],
                 request=request.model_input(),binding=binding)
        label=copy.deepcopy(target)
        label.update(id=uid,source_id=item['id'],variant=variant,
            targets=[q.values.index(v) for q,v in zip(request.questions,target['target_values'],strict=True)],
            request_sha256=content_hash(row),request_weight=weight/len(variants),
            original_request_weight=weight,augmentation_views=len(variants))
        expanded.append((row,label))
    if not math.isclose(sum(label['request_weight'] for _,label in expanded),weight,rel_tol=1e-12):
        raise ValueError('source weight budget changed')
    return expanded
