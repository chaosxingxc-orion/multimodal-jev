"""Stage1 membership-logit InfoNCE. A positive is never another positive's negative."""
from dataclasses import dataclass
import math

KIND = 'card_bidirectional_infonce_v1'
SPEECH_KIND = 'card_and_speech_infonce_v1'
CARD_KINDS = (KIND, SPEECH_KIND)
TRIPLET_KINDS = ('intent_triplet_infonce_v1', *CARD_KINDS)


@dataclass(frozen=True)
class SpeechObjective:
    fixed_card_temperature: float
    speech_temperature: float
    speech_weight: float
    probability_margin: float
    margin_weight: float

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError('explicit speech contrast temperatures, margin and weights required')
        if any(type(x) not in (float, int) or not math.isfinite(x) or x <= 0 for x in value.values()):
            raise ValueError('positive finite speech objective settings required')
        if value['probability_margin'] >= 1:
            raise ValueError('probability margin must be less than one')
        return cls(**value)


@dataclass(frozen=True)
class Directions:
    positives: int  # additional same-class recordings, excluding the anchor
    cards: int
    recordings: int
    replacement: bool
    temperature: float
    audio_weight: float   # fixed audio, vary card
    card_weight: float    # fixed card, vary audio

    speech: SpeechObjective | None = None

    @classmethod
    def from_objective(cls, objective):
        keys = {'kind', 'temperature', 'negative_cards_per_anchor', 'fixed_audio_weight',
                'fixed_card_weight', 'probability_coefficient', 'positive_recordings_per_anchor',
                'negative_recordings_per_anchor', 'sample_with_replacement'}
        speech = None
        if objective.get('kind') == SPEECH_KIND:
            keys.add('speech')
            speech = SpeechObjective.parse(objective.get('speech'))
        if (set(objective) != keys or objective['kind'] not in CARD_KINDS or
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
        if speech and min(a, c) <= 0:
            raise ValueError('speech objective needs both membership directions')
        return cls(*counts, replacement, t, a, c, speech)

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
                *(('jev/fixed_card_nce',) if self.card_weight else ()),
                *(('jev/speech_nce', 'jev/probability_margin') if self.speech else ()))

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


def components(logits, directions, speech_vectors=None):
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
        card_z = logits.float() / directions.speech.fixed_card_temperature if directions.speech else z
        negatives = torch.logsumexp(card_z[:, split:directions.observed], dim=-1, keepdim=True)
        positives = torch.cat((card_z[:, :1], card_z[:, directions.observed:]), dim=1)
        terms.append(F.softplus(negatives - positives).mean(dim=1))
        weights.append(directions.card_weight)
    if directions.speech:
        cfg = directions.speech
        if (speech_vectors is None or speech_vectors.ndim != 3 or
                speech_vectors.shape[:2] != logits.shape or not speech_vectors.shape[-1]):
            raise ValueError('one pre-card speech representation per scored pair required')
        # Pair order is anchor, wrong cards on the anchor, wrong recordings, extra positives.
        vectors = F.normalize(speech_vectors.float(), dim=-1)
        similarities = (vectors[:, :1] * vectors[:, split:]).sum(-1) / cfg.speech_temperature
        negatives, positives = similarities.split((directions.recordings, directions.positives), dim=1)
        speech_nce = F.softplus(negatives.logsumexp(1, keepdim=True) - positives).mean(1)
        p = logits.float().sigmoid()  # deployed member probability, never sigmoid(z / temperature)
        true = torch.cat((p[:, :1], p[:, directions.observed:]), dim=1)
        card_gap = p[:, 0] - p[:, 1:split].amax(1)
        recording_gap = true - p[:, split:directions.observed].amax(1, keepdim=True)
        penalty = (directions.audio_weight * (cfg.probability_margin - card_gap).relu()
                   + directions.card_weight * (cfg.probability_margin - recording_gap).relu().mean(1))
        terms.extend((speech_nce, penalty))
        weights.extend((cfg.speech_weight, cfg.margin_weight))
    elif speech_vectors is not None:
        raise ValueError('unexpected speech features for membership-only objective')
    return torch.stack([sum(w*term for w, term in zip(weights, terms, strict=True)), *terms], dim=1)


def bind_sample_counts(config):
    """Transfer public experiment counts into the single frozen objective."""
    names = {'positive_recordings': 'positive_recordings_per_anchor',
             'negative_cards': 'negative_cards_per_anchor',
             'negative_recordings': 'negative_recordings_per_anchor'}
    counts = config['contrast_samples']
    objective = config['window']['triplet_objective']
    if (not isinstance(counts, dict) or counts.keys() != names.keys()
            or any(type(value) is not int or value < 1 for value in counts.values())
            or any(key in objective for key in names.values())):
        raise ValueError('declare positive contrast counts once at experiment level')
    bound = dict(objective, **{target: counts[key] for key,target in names.items()})
    Directions.from_objective(bound)
    config['window']['triplet_objective'] = bound
