"""Measure complete, identical input blocks under the two inference paths."""
import copy
import hashlib
import json

from multimodal_jev.card_score_latency import ScoreLatency
from multimodal_jev.ordered_prefetch import OrderedPrefetch


def compare_block(scorer, model, device, provider, policy, block, encode_one,
                  *, workers, prefetch, aligned, reference, rounds, swap):
    if type(rounds) is not int or rounds < 2 or rounds % 2:
        raise ValueError('balanced repeated timing required')
    def run(mode):
        local = copy.copy(scorer)
        local.mode = mode
        timer = ScoreLatency(len(local.candidates))
        with OrderedPrefetch(encode_one, block['ids'], workers=workers, ahead=prefetch) as stream:
            result = local.score(model, device, provider, policy, stream, check_batch=False, latency=timer)
        times = timer.result()
        if (result['policy_id'] != policy or [r['id'] for r in result['records']] != block['ids']
                or any(len(r['prob']) != n for r,n in zip(result['records'],block['widths'],strict=True))):
            raise ValueError('timed path changed block membership')
        return result, times

    ref, _ = run(reference)
    reference_sha = hashlib.sha256(json.dumps(ref, separators=(',', ':'),allow_nan=False).encode()).hexdigest()
    trials = []
    # Untimed warm-up for both paths with identical input and prepared artifacts.
    for mode in ('bounded_batch', aligned):
        run(mode)
    for i in range(rounds):
        arms = [('standard', 'bounded_batch'), ('candidate', aligned)]
        if (i + swap) % 2:
            arms.reverse()
        trial = {}
        for arm, mode in arms:
            result, times = run(mode)
            differences = sum(a != b for x,y in zip(result['records'],ref['records'],strict=True)
                              for a,b in zip(x['prob'],y['prob'],strict=True))
            trial[arm] = dict(**times, differences=differences)
        trials.append(trial)
    return dict(block=block['key'], reference=reference, reference_sha256=reference_sha, mode=aligned,
                recordings=len(block['ids']), pairs=sum(block['widths']), trials=trials)
