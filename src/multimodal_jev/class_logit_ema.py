"""Online class-conditional logit prototypes for hard-negative sampling."""
import math

import torch
import numpy as np


class ClassLogitEMA:
    def __init__(self, labels, *, decay, temperature, uniform_mix, min_shared_coordinates,
                 max_staleness_updates):
        if (len(labels) < 2 or len(set(labels)) != len(labels)
                or not 0. <= decay < 1. or temperature <= 0.
                or not 0. < uniform_mix <= 1.
                or type(min_shared_coordinates) is not int or not 1 <= min_shared_coordinates <= len(labels)
                or type(max_staleness_updates) is not int or max_staleness_updates < 0):
            raise ValueError('invalid class-logit EMA configuration')
        if not all(math.isfinite(x) for x in (decay, temperature, uniform_mix)):
            raise ValueError('EMA parameters must be finite')
        self.labels = tuple(labels)
        self.decay = float(decay)
        self.temperature = float(temperature)
        self.uniform_mix = float(uniform_mix)
        self.min_shared_coordinates = min_shared_coordinates
        self.max_staleness_updates = max_staleness_updates
        self.label_index = {label: i for i, label in enumerate(labels)}
        size = len(labels)
        self.values = torch.zeros((size, size), dtype=torch.float32)
        self.observed = torch.zeros((size, size), dtype=torch.bool)
        self.observation_counts = torch.zeros((size, size), dtype=torch.long)
        self.updates = 0

    def update(self, sums, counts):
        """Consume CPU views of the already globally reduced statistics."""
        if sums.shape != self.values.shape or counts.shape != sums.shape:
            raise ValueError('class-logit update shape differs')
        sums, counts = sums.numpy(), counts.numpy()
        mask = counts > 0
        if not np.isfinite(sums).all() or not mask.any():
            raise ValueError('invalid global class-logit statistics')
        values = self.values.numpy()
        observed = self.observed.numpy()
        means = sums[mask] / counts[mask]
        values[mask] = np.where(observed[mask],
            self.decay * values[mask] + (1 - self.decay) * means, means)
        observed[mask] = True
        self.observation_counts.numpy()[mask] += counts[mask].astype(np.int64)
        self.updates += 1

    def probabilities(self):
        # Each cosine uses the intersection of the two observed coordinate sets.
        v = self.values.numpy().astype(np.float64)
        mask = self.observed.numpy().astype(np.float64)
        def cross(left, right):
            return np.einsum('ik,jk->ij', left, right)
        norms = cross(v*v, mask)
        denom = np.sqrt(norms * norms.T)
        valid = (cross(mask, mask) >= self.min_shared_coordinates) & (denom > 0)
        sim = np.divide(cross(v, v), denom, where=valid, out=np.zeros_like(denom))
        np.fill_diagonal(sim, -np.inf)
        weights = np.exp((sim - sim.max(axis=1, keepdims=True)) / self.temperature)
        q = (1 - self.uniform_mix) * weights / weights.sum(axis=1, keepdims=True)
        q += self.uniform_mix / (len(self.labels) - 1)
        np.fill_diagonal(q, 0)
        return q

    def distribution(self):
        return {label: {other: prob for other, prob in zip(self.labels, row) if other != label}
                for label, row in zip(self.labels, self.probabilities().tolist())}

    def receipt(self, config):
        from .span_request_identity import tensor_identity
        return dict(kind='class-logit-ema-v1', updates=self.updates, labels=list(self.labels),
                    config=config, values_sha256=tensor_identity(self.values)['sha256'],
                    observed_sha256=tensor_identity(self.observed)['sha256'],
                    counts_sha256=tensor_identity(self.observation_counts)['sha256'],
                    anchor_counts=self.observation_counts.diag().tolist(),
                    distribution_sha256=tensor_identity(torch.from_numpy(self.probabilities()))['sha256'])

    def state_dict(self):
        return dict(schema='class-logit-ema-v1', labels=list(self.labels), decay=self.decay,
            temperature=self.temperature, uniform_mix=self.uniform_mix,
            min_shared_coordinates=self.min_shared_coordinates, updates=self.updates,
            max_staleness_updates=self.max_staleness_updates,
            values=self.values.detach().cpu().clone(), observed=self.observed.detach().cpu().clone(),
            observation_counts=self.observation_counts.detach().cpu().clone())

    @torch.no_grad()
    def load_state_dict(self, state):
        expected = (state.get('schema') == 'class-logit-ema-v1'
            and state.get('labels') == list(self.labels)
            and state.get('decay') == self.decay and state.get('temperature') == self.temperature
            and state.get('uniform_mix') == self.uniform_mix
            and state.get('min_shared_coordinates') == self.min_shared_coordinates
            and state.get('max_staleness_updates') == self.max_staleness_updates)
        values, observed, observation_counts = state.get('values'), state.get('observed'), state.get('observation_counts')
        if (not expected or not isinstance(values, torch.Tensor) or not isinstance(observed, torch.Tensor)
                or values.shape != self.values.shape or observed.shape != self.observed.shape
                or values.dtype != torch.float32 or observed.dtype != torch.bool
                or not isinstance(observation_counts, torch.Tensor)
                or observation_counts.shape != self.observation_counts.shape
                or observation_counts.dtype != torch.long or (observation_counts < 0).any()
                or not torch.isfinite(values).all() or type(state.get('updates')) is not int
                or not torch.equal(observed, observation_counts > 0) or (values[~observed] != 0).any()
                or state['updates'] < 0):
            raise ValueError('class-logit EMA checkpoint differs')
        self.values.copy_(values)
        self.observed.copy_(observed)
        self.observation_counts.copy_(observation_counts)
        self.updates = state['updates']


def pack_observations(output, logits, groups, label_index):
    """Write this rank's disjoint slots. Padding has coordinate -1."""
    size = len(label_index)
    indices = []
    for _, weight, _, coordinates in groups:
        gold = label_index[coordinates[0]]
        if weight == 1:
            indices.extend(gold * size + label_index[label] for label in coordinates)
        elif weight == 0:
            indices.extend([-1] * len(coordinates))
        else:
            raise ValueError('expected real or padding anchor')
    rows = logits.detach().reshape(len(groups), -1)
    output[:, :, 0].copy_(torch.tensor(indices, dtype=torch.float32, device=output.device).reshape_as(output[:, :, 0]))
    output[:, :1, 1].copy_(rows[:, :1])
    output[:, 1:, 1].copy_(rows[:, 3:])


def global_tables(slots, size):
    """The existing reduction assembled all ranks. Aggregate in fixed rank order."""
    coords, values = slots.numpy().reshape(-1, 2).T
    coords = coords.astype(np.int64)
    real = coords >= 0
    coords = coords[real]
    def count(weights=None):
        arr = np.bincount(coords, weights=weights, minlength=size*size)
        return torch.from_numpy(arr.reshape(size, size))
    return count(values[real]), count()
