"""Summarize a bounded py-spy speedscope capture without model analysis.

Weights are sampled active-thread time, not exclusive wall-clock attribution.
This tool never attaches to a process, schedules work or changes a live job.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

from multimodal_jev.baseline import file_hash
from multimodal_jev.local_checkpoint import commit_json
from multimodal_jev.local_config import load_config


def summarize(data, *, top=10):
    if type(top) is not int or not 1 <= top <= 100:
        raise ValueError('bounded top count required')
    frames = data['shared']['frames']
    result = []
    for profile in data['profiles']:
        if profile['type'] != 'sampled' or profile['unit'] != 'seconds':
            raise ValueError('sampled seconds profile required')
        leaf = Counter(); callers = Counter(); total = 0.
        for stack, weight in zip(profile['samples'], profile['weights'], strict=True):
            if type(weight) not in (int, float) or not 0 <= weight < float('inf'):
                raise ValueError('finite nonnegative sample weight required')
            if any(type(i) is not int or not 0 <= i < len(frames) for i in stack):
                raise ValueError('valid frame indices required')
            total += weight
            if stack:
                leaf[stack[-1]] += weight
                for index in set(stack):
                    callers[index] += weight
        def entries(counter):
            return [dict(frame=frames[i], sampled_seconds=value,
                         fraction_of_thread_samples=value/total if total else 0.)
                    for i, value in counter.most_common(top)]
        result.append(dict(thread=profile['name'], sampled_seconds=total,
                           leaf=entries(leaf), inclusive=entries(callers)))
    return dict(status='cpu_profile_summarized', threads=result, model_calls=0,
                scope='Sampled active-thread weights; inclusive frames overlap; not exclusive wall-clock costs')


def run(config, provenance):
    if file_hash(config['profile']) != config['profile_sha256']:
        raise ValueError('profile content changed')
    result = summarize(json.loads(Path(config['profile']).read_text()), top=config['top'])
    result.update(profile_sha256=config['profile_sha256'], provenance=provenance)
    path = Path(config['output'])
    if path.exists():
        if json.loads(path.read_text()) != result:
            raise ValueError('immutable profile summary differs')
    else:
        commit_json(path, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--local', required=True)
    args = p.parse_args()
    c, provenance = load_config(args.config, args.local)
    report = run(c, provenance)
    print(json.dumps(dict(status=report['status'], threads=len(report['threads']))))
