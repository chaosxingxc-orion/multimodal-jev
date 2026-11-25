"""Stage1 membership-logit InfoNCE. A positive is never another positive's negative."""
from dataclasses import dataclass
import math

KIND = 'card_bidirectional_infonce_v1'
TRIPLET_KINDS = ('intent_triplet_infonce_v1', KIND)


@dataclass(frozen=True)
class Directions:
    positives: int  # additional same-class recordings, excluding the anchor
    cards: int
    recordings: int
    replacement: bool
    temperature: float
    audio_weight: float   # fixed audio, vary card
    card_weight: float    # fixed card, vary audio

    @classmethod
    def from_objective(cls, objective):
        keys = {'kind', 'temperature', 'negative_cards_per_anchor', 'fixed_audio_weight',
                'fixed_card_weight', 'probability_coefficient', 'positive_recordings_per_anchor',
                'negative_recordings_per_anchor', 'sample_with_replacement'}
        if (set(objective) != keys or objective['kind'] != KIND or
                objective['probability_coefficient'] != 0):
            raise ValueError('expected the explicit membership-logit contrastive objective')
        counts = tuple(objective[key] for key in ('positive_recordings_per_anchor', 'negative_cards_per_anchor',
                                                  'negative_recordings_per_anchor'))
        replacement = objective['sample_with_replacement']
        t, a, c = (objective[name] for name in ('temperature', 'fixed_audio_weight', 'fixed_card_weight'))
        if (any(type(n) is not int or n < 1 for n in counts) or type(replacement) is not bool or
                any(type(x) not in (int, float) or not math.isfinite(x) for x in (t, a, c)) or
                t <= 0 or min(a, c) < 0 or not math.isclose(a+c, 1.)):
            raise ValueError('positive K and temperature; nonnegative direction weights must sum to one')
        return cls(*counts, replacement, t, a, c)

    @property
    def anchor_end(self):
        return 1 + (self.cards if self.audio_weight else 0)

    @property
    def width(self):
        return self.anchor_end + (self.positives + self.recordings if self.card_weight else 0)

    @property
    def observed(self):
        # Only one diagonal observation per real anchor. Extra positives participate in loss only.
        return self.anchor_end + (self.recordings if self.card_weight else 0)

    @property
    def metrics(self):
        return ('actor/pg_loss', *(('jev/fixed_audio_nce',) if self.audio_weight else ()),
                *(('jev/fixed_card_nce',) if self.card_weight else ()))

    def pairs(self, anchor, gold, positives, wrong, negatives):
        # One template assignment for the entire group, including its few-shot panel.
        pairs = [(anchor, gold, anchor)]
        coords = [(gold, gold)]
        if self.audio_weight:
            pairs.extend((anchor, label, anchor) for label in wrong)
            coords.extend((gold, label) for label in wrong)
        if self.card_weight:
            pairs.extend((uid, gold, anchor) for _, uid in negatives)
            pairs.extend((uid, gold, anchor) for uid in positives)
            coords.extend((label, gold) for label, _ in negatives)
        return pairs, coords


def components(logits, directions):
    """Columns: A/a, [A/c_i], [C_i/a, B_j/a]. Return per-anchor loss terms."""
    import torch
    import torch.nn.functional as F
    if logits.ndim != 2 or logits.shape[1] != directions.width:
        raise ValueError('one membership logit required per rendered pair')
    z = logits.float() / directions.temperature
    split = directions.anchor_end
    terms, weights = [], []
    if directions.audio_weight:
        terms.append(-F.log_softmax(z[:, :split], dim=-1)[:, 0])
        weights.append(directions.audio_weight)
    if directions.card_weight:
        # Each positive has its own denominator containing only it and the K negatives.
        negatives = torch.logsumexp(z[:, split:directions.observed], dim=-1, keepdim=True)
        positives = torch.cat((z[:, :1], z[:, directions.observed:]), dim=1)
        terms.append(F.softplus(negatives - positives).mean(dim=1))
        weights.append(directions.card_weight)
    return torch.stack([sum(w*term for w, term in zip(weights, terms, strict=True)), *terms], dim=1)
