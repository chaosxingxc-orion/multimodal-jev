"""Recreate the seed-bound two-epoch recording schedule and zero-weight tails."""
from collections import Counter
import random

from .context_replica_schedule import rebatch
from .slurp_training import content_hash
from .span_class_spread import class_spread_indices


def schedule(population, config, seed):
    epochs = []
    for epoch in range(config['epochs']):
        ids = sorted(population)
        random.Random(f'{seed}|source|{epoch}').shuffle(ids)
        order = class_spread_indices([population[i]['intent'] for i in ids],seed=seed,epoch=epoch)
        slots = [dict(id=ids[i],weight=1.,padding=False) for i in order]
        epochs.append(dict(epoch=epoch,index=0,slots=slots))
    batches = rebatch(epochs,list(population),config)
    return batches,dict(seed=seed,epochs=config['epochs'],updates=len(batches),
                        batches_sha256=content_hash(batches),population=len(population))
