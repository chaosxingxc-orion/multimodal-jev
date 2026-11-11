"""Audit actual exposure equality across completed transfer trajectories."""
import math


def compare_training_exposure(branches):
    """Compare actual step records; do not infer compute parity from step count.

    Each entry supplies its study hash, initial policy, step configs/reports and
    observed execution duration. Callers authenticate these against checkpoints.
    """
    if set(branches)!={'sft','rl','conditional'}:
        raise ValueError('three prespecified methods required')
    reference=branches['sft'];common=None;budgets={}
    for method,branch in branches.items():
        if (branch['study_sha256']!=reference['study_sha256']
                or branch['initial_policy_id']!=reference['initial_policy_id']):
            raise ValueError('different study or initializer')
        steps=branch['steps']
        if not steps:raise ValueError('actual completed updates required')
        exposure=[]
        for i,step in enumerate(steps):
            c=step['config'];r=step['report']
            if (c['method']!=method or c['batch_index']!=i or r['completed_updates']!=i+1
                    or r['optimizer_updates']!=1 or r['method']!=method
                    or r['training_requests']!=len(c['request_ids'])
                    or r['optimizer_start']!=('fresh' if i==0 else 'resume')
                    or len(c['request_ids'])!=len(set(c['request_ids']))):
                raise ValueError('nonconsecutive actual method/update/exposure')
            for key in ('input_tokens','audio_seconds_with_repetition','unique_recordings'):
                if not isinstance(r[key],(int,float)) or not math.isfinite(r[key]) or r[key]<=0:
                    raise ValueError('positive measured training cost required')
            exposure.append(dict(request_ids=c['request_ids'],seed=c['seed'],arm=c['arm'],
                auxiliary_weight=c['auxiliary_weight'],lr=c['lr'],samples_per_request=c['samples_per_request'],
                input_tokens=r['input_tokens'],audio_seconds=r['audio_seconds_with_repetition'],
                unique_recordings=r['unique_recordings']))
        if common is None:common=exposure
        elif exposure!=common:raise ValueError('training inputs, order, objective weight or measured exposure differ')
        seconds=branch['wall_seconds_including_gates']
        if not math.isfinite(seconds) or seconds<=0:raise ValueError('positive measured elapsed time required')
        budgets[method]=dict(updates=len(steps),requests=sum(len(x['request_ids']) for x in exposure),
            input_tokens=sum(x['input_tokens'] for x in exposure),
            audio_seconds_with_repetition=sum(x['audio_seconds'] for x in exposure),
            wall_seconds_including_gates=seconds,
            policy_gradient_samples_used=sum(len(s['config']['request_ids'])*s['config']['samples_per_request'] for s in steps) if method=='rl' else 0,
            exact_support_integration=method=='conditional')
    return dict(passed=True,matched_updates_and_training_exposure=True,budgets=budgets,
        scope='Actual per-update request order and recorded exposure match; device time includes initialization and parity gates, not equal FLOPs or pure training throughput')
