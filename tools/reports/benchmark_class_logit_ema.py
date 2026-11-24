"""Measure host sampler work on a synthetic roster with the registered Stage1 geometry."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from multimodal_jev.class_logit_ema import ClassLogitEMA, global_tables
from multimodal_jev.intent_triplet_sampling import AnchorSampler


def measure(window, population, classes, iterations, warmup, seed):
    labels = [str(i) for i in range(classes)]
    k = window['triplet_objective']['negative_cards_per_anchor']
    micro, world = window['execution_microbatch'], window['training_npus']
    ema = ClassLogitEMA(labels, **window['class_logit_ema'])
    records = {i: dict(id=i, intent=labels[i % classes], role='learning', official_split='train')
               for i in range(population)}
    start = perf_counter()
    sampler = AnchorSampler(records, labels, k, seed)
    setup = perf_counter() - start
    sample = np.random.default_rng(seed)
    ms = []
    for step in range(warmup + iterations):
        # One global batch of representative real anchor observations. No model or audio I/O.
        probes = np.empty((world * micro, k+1, 2), dtype=np.float32)
        for row in probes:
            gold = int(sample.integers(classes))
            other = sample.choice(classes-1,k,replace=False)
            row[:,0] = gold*classes + np.r_[gold, other+(other >= gold)]
        probes[:,:,1] = sample.standard_normal(probes.shape[:2])
        probes = torch.from_numpy(probes)
        start = perf_counter()
        ema.update(*global_tables(probes, classes))
        q = ema.distribution()
        for i in range(micro):
            sampler.draw((step*micro+i) % population, 0, step, q)
        if step >= warmup:
            ms.append(1000*(perf_counter()-start))
    return dict(synthetic_population=population, classes=classes, updates=iterations, warmup=warmup,
                index_build_ms=round(1000*setup,2), median_ms=round(float(np.median(ms)),3),
                p95_ms=round(float(np.percentile(ms,95)),3), npu_executed=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path)
    for key in ('population','classes','iterations','warmup'):
        p.add_argument('--'+key, type=int, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_bytes())
    print(json.dumps(measure(cfg['window'],args.population,args.classes,args.iterations,args.warmup,cfg['seed'])))
